import os
import nemo
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import torch
from nemo.collections.asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
from functools import partial


nf = nemo.core.NeuralModuleFactory(
    log_dir='QuartzNet15x5',
    create_tb_writer=True)

logger = nemo.logging
tb_writer = nf.tb_writer

config_path = 'quartznet15x5.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
labels = params['labels']

data_dir = "."

if not os.path.exists(data_dir+'/quartz_checkpoints'):
    os.makedirs(data_dir+'/quartz_checkpoints')
    
train_manifest = data_dir + "/manifests/train_manifest.json"

valid_with_real_manifest = data_dir + \
    "/manifests/valid_with_read_labels_manifest.json"

valid_with_corrected_manifest = data_dir + \
    "/manifests/valid_with_corrected_labels_manifest.json"

data_layer_train = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer",
    overwrite_params={"manifest_filepath": train_manifest}
)

data_layer_eval_real = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer",
    overwrite_params={"manifest_filepath": valid_with_real_manifest}
)

data_layer_eval_corrected = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer",
    overwrite_params={"manifest_filepath": valid_with_corrected_manifest}
)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
    config_path, "AudioToMelSpectrogramPreprocessor"
)

spec_augment = nemo_asr.SpectrogramAugmentation.import_from_config(
    config_path, "SpectrogramAugmentation"
)

encoder = nemo_asr.JasperEncoder.import_from_config(
    config_path, "JasperEncoder"
)

decoder = nemo_asr.JasperDecoderForCTC.import_from_config(
    config_path, "JasperDecoderForCTC",
    overwrite_params={"num_classes": len(labels)}
)

#Load pretrained

encoder.restore_from(
    "pretrained_models/JasperEncoder-STEP-247400.pt")
decoder.restore_from(
    "pretrained_models/JasperDecoderForCTC-STEP-247400.pt")

ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

# Train DAG
audio_signal, audio_signal_len, transcript, transcript_len = data_layer_train()

processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)

encoded, encoded_len = encoder(
    audio_signal=processed_signal,
    length=processed_signal_len)

log_probs = decoder(encoder_output=encoded)
predictions = greedy_decoder(log_probs=log_probs)  # Training predictions
loss = ctc_loss(
    log_probs=log_probs,
    targets=transcript,
    input_length=encoded_len,
    target_length=transcript_len)

#VAL DAG-1
audio_signal_v1, audio_signal_len_v1, transcript_v1, transcript_len_v1 = data_layer_eval_real()

processed_signal_v1, processed_signal_len_v1 = data_preprocessor(
    input_signal=audio_signal_v1, length=audio_signal_len_v1)
# no Aug
encoded_v1, encoded_len_v1 = encoder(
    audio_signal=processed_signal_v1, length=processed_signal_len_v1)
log_probs_v1 = decoder(encoder_output=encoded_v1)
predictions_v1 = greedy_decoder(log_probs=log_probs_v1)

loss_eval_real = ctc_loss(
    log_probs=log_probs_v1, targets=transcript_v1,
    input_length=encoded_len_v1, target_length=transcript_len_v1)

#VAL DAG-2
audio_signal_v2, audio_signal_len_v2, transcript_v2, transcript_len_v2 = data_layer_eval_corrected()

processed_signal_v2, processed_signal_len_v2 = data_preprocessor(
    input_signal=audio_signal_v2, length=audio_signal_len_v2)
# no Aug
encoded_v2, encoded_len_v2 = encoder(
    audio_signal=processed_signal_v2, length=processed_signal_len_v2)
log_probs_v2 = decoder(encoder_output=encoded_v2)
predictions_v2 = greedy_decoder(log_probs=log_probs_v2)

loss_eval_corrected = ctc_loss(
    log_probs=log_probs_v2, targets=transcript_v2,
    input_length=encoded_len_v2, target_length=transcript_len_v2)

train_callback = nemo.core.SimpleLossLoggerCallback(
    tb_writer=tb_writer,
    tensors=[loss, predictions, transcript, transcript_len],
    print_func=partial(
        monitor_asr_train_progress,
        labels=labels
    ))

saver_callback = nemo.core.CheckpointCallback(
    folder=data_dir+'/quartz_checkpoints',
    step_freq=1000)

eval_real_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_eval_real, predictions_v1,
                  transcript_v1, transcript_len_v1],
    user_iter_callback=partial(
        process_evaluation_batch,
        labels=labels
    ),
    user_epochs_done_callback=partial(
        process_evaluation_epoch, tag="EVAL ON REAL LABELS"
    ),
    eval_step=500,
    tb_writer=tb_writer)

eval_corrected_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_eval_corrected, predictions_v2,
                  transcript_v2, transcript_len_v2],
    user_iter_callback=partial(
        process_evaluation_batch,
        labels=labels
    ),
    user_epochs_done_callback=partial(
        process_evaluation_epoch, tag="EVAL ON CORRECTED LABELS"
    ),
    eval_step=500,
    tb_writer=tb_writer)

nf.train(
    tensors_to_optimize=[loss],
    callbacks=[train_callback, eval_real_callback,
               eval_corrected_callback, saver_callback],
    optimizer="novograd",
    optimization_params={
        "num_epochs": 200, "lr": 1e-3, "weight_decay": 1e-4
    }
)
