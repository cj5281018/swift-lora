import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import InferEngine, InferRequest, RequestConfig, TransformersEngine, get_template

# 输入推理路径
last_model_checkpoint = 'output/v9-20260628-005031/checkpoint-94'

# 模型
model_id_or_path = 'models/Qwen2.5-7B-Instruct'  # model_id or model_path
system = 'You are a helpful assistant.'

# 生成参数
max_new_tokens = 512
temperature = 0
stream = True

engine = TransformersEngine(model_id_or_path, adapters=[last_model_checkpoint])
template = get_template(engine.processor, default_system=system)
# 这里对推理引擎的默认template进行修改，也可以在`engine.infer`时进行传入
engine.default_template = template

query_list = [
    'who are you?',
    "晚上睡不着觉怎么办？",
    '你是谁训练的？',
]

def infer_stream(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=True)
    gen_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end='', flush=True)
    print()

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')

infer_func = infer_stream if stream else infer
for query in query_list:
    infer_func(engine, InferRequest(messages=[{'role': 'user', 'content': query}]))
    print('-' * 50)