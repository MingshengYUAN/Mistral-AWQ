from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request
import time
from flasgger import Swagger
from awq import AutoAWQForCausalLM

device = "cuda:0" # the device to load the model onto

app = Flask(__name__)
templete = {
  "swagger": "2.0",
  "info": {
    "title": "Mistral AWQ Model API",
    "description": "Mistral AWQ Model API",
    "version": "0.0.1"
  },
  "tags": [
    {
      "name": "Input prompt",
      "description": "Input prompt"
    }
  ],
  "paths": {
    "/generate": {
      "post": {
        "tags": [
          "generate"
        ],
        "summary": "Generate Output",
        "description": "",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "prompt",
            "required": True,
            "schema": {
              "$ref": "#/definitions/prompt"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Response *** The <Answer:> or the prompt in the response is generate by the model, not manually added !! *** ",
            "schema": {
              "$ref": "#/definitions/response"
            }
          }
        }
      }
    }
  },
  "definitions": {
    "prompt": {
      "type": "object",
      "required": [
        "prompt"
      ],
      "properties": {
        "prompt": {
          "type": "string",
          "items": {
            "type": "string"
          },
          "example": "What is your name?"
        }
      }
    },
    "response": {
      "type": "object",
      "properties": {
        "response": {
          "items": {
            "type": "string"
          },
          "example": "I'm ....."
        },
        "status": {
          "items": {
            "type": "string"
          },
          "example": "success"
        },
        "running_time": {
          "items": {
            "type": "number"
          },
          "example": "0.325542"
        }
      }
    },
    "ApiResponse": {
      "type": "object",
      "properties": {
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "type": {
          "type": "string"
        },
        "message": {
          "type": "string"
        }
      }
    }
  }
}
swagger = Swagger(app, template=templete)

model = AutoAWQForCausalLM.from_quantized("/data-gpu02/home/mingsheng.yuan/yma/LLM/LLAMA2/Mistral/7B-AWQ-Instruct-v0.1", fuse_layers=True, trust_remote_code=False, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained("/data-gpu02/home/mingsheng.yuan/yma/LLM/LLAMA2/Mistral//7B-AWQ-Instruct-v0.1", trust_remote_code=False)

model.to(device)

@app.route("/generate", methods=['POST'])
def generate():
    start = time.time()
    data = request.get_json()
    print(data)
    encodeds = tokenizer(data['prompt'], return_tensors="pt").input_ids
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=512, top_p=0.9, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    try:
      response = decoded[0].split("A: ")[1]
      ret = {"response": response.strip("</s>"), "status": 'success', "running_time": float(time.time() - start)}
    except:
      ret = {"response": decoded[0], "status": 'success', "running_time": float(time.time() - start)}
    return ret

if __name__=="__main__":
  app.run(port=3092, host="0.0.0.0", debug=False)