FROM tensorflow/serving:2.14.1

COPY ./serving_model_dir /models/heart-disease-model
COPY ./monitoring /model_config

ENV MODEL_NAME=heart-disease-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash \n\n\ 
ulimit -s 2048 \n\ 
env \n\ 
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \ 
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \ 
--monitoring_config_file=${MONITORING_CONFIG} \ 
--file_system_poll_wait_seconds=0 \ 
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
