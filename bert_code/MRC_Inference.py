
# MRC model API for Dev Hack

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import flask

import run_squad as mainfile

app = flask.Flask(__name__)
#MODELDIR = './output/squad_base'

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (start_logits, end_logits) = mainfile.create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=None)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=None)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn
        

        

def init():
    global mrc_inference_config, tokenizer, estimator
    with open("./MRC_Inference_Config.json", "r") as config_file:          
        mrc_inference_config = json.load(config_file)  
    bert_config = modeling.BertConfig.from_json_file(mrc_inference_config["bert_config_file"])    
    tf.gfile.MakeDirs(mrc_inference_config["output_dir"])
    tokenizer = tokenization.FullTokenizer(vocab_file=mrc_inference_config["vocab_file"], do_lower_case=True)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=mrc_inference_config["model_path"],
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host))
        
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=1000000,
        num_warmup_steps=100,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=32,
      predict_batch_size=8)


# API for prediction
@app.route("/mrc", methods=["POST"])
def mrc():
    data_from_post = getData()
    data = preprocess_data(data_from_post)
    
    eval_writer = mainfile.FeatureWriter(
        filename=os.path.join(mrc_inference_config["output_dir"], "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    mainfile.convert_examples_to_features(
        examples=data,
        tokenizer=tokenizer,
        max_seq_length=mrc_inference_config["max_seq_length"],
        doc_stride=mrc_inference_config["doc_stride"],
        max_query_length=mrc_inference_config["max_query_length"],
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    all_results = []

    predict_input_fn = mainfile.input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=mrc_inference_config["max_seq_length"],
        is_training=False,
        drop_remainder=False)
    
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):      
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            mainfile.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    answer = mainfile.write_predictions(data,eval_features,all_results,20,mrc_inference_config["max_answer_length"],True,None,None,None)
    return sendResponse({"Answer": answer.get(data_from_post.get("qas_id"))})


# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

def preprocess_data(raw_data):
    examples = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    paragraph_text = raw_data["context"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    qas_id = raw_data["qas_id"]
    question_text = raw_data["question"]
    start_position = None
    end_position = None
    orig_answer_text = None
    is_impossible = False

    data = mainfile.SquadExample(
        qas_id=qas_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position,
        is_impossible=is_impossible)
    examples.append(data)
    return examples

def getData():
    data = {}
    data["qas_id"] = flask.request.form.get("qas_id")
    data["question"] = flask.request.form.get("question")
    data["context"] = flask.request.form.get("context")
    return data

if __name__ == "__main__":
    init()
    app.run(threaded=True)

