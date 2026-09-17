FROM tensorflow/serving:latest

COPY ./serving_model_dir /models/heart-disease-model
COPY ./monitoring /model_config

ENV MODEL_NAME=heart-disease-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port= \
--model_name= --model_base_path=/ \
--monitoring_config_file= \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
