import abc
import base64
import httpx
import os
import subprocess
import tempfile

# from google import genai
# from openai import OpenAI
import botocore
import boto3

from io import BytesIO
from pypdf import PdfReader, PdfWriter

from api_key import GEMINI_API_KEY, OPENAI_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY

class LLMWrapper(abc.ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abc.abstractmethod
    def _send(self, message):
        pass
    
    @abc.abstractmethod
    def _stats(self):
        pass
    
    def send_pdf(self, message, pdf_url):
        pdf_data = None
        if pdf_url:
            pdf_data = httpx.get(pdf_url).content

            # optional lines to make PDFs smaller (i.e. cheaper to input)
            pdf_data = self.clip_pdf_bytes(pdf_data, max_pages=15, debug=True)
            if len(pdf_data) > 1024 * 1024 * 5:
                print(f"\tBefore compression: {len(pdf_data)/ 1e6: .2f}MB")
                pdf_data = self.compress_pdf_bytes_gs(pdf_data)
                print(f"\tAfter compression: {len(pdf_data)/ 1e6: .2f}MB")

        return self._send(message, pdf_data)

    def clip_pdf_bytes(self, pdf_bytes, max_pages=15, debug=False):
        reader = PdfReader(BytesIO(pdf_bytes))
        writer = PdfWriter()

        for i in range(min(max_pages, len(reader.pages))):
            writer.add_page(reader.pages[i])

        out = BytesIO()
        writer.write(out)
        clipped_bytes = out.getvalue()
        if debug:
            pid = os.getpid()
            out_path = f"/tmp/llm_clip_{pid}_{max_pages}.pdf"
            with open(out_path, "wb") as f:
                f.write(clipped_bytes)
            print(f"\t[DEBUG] clipped PDF written to {out_path}")
        return clipped_bytes
    

    def compress_pdf_bytes_gs(self, pdf_bytes, pdfsettings="/ebook", compatibility_level="1.4", timeout_s=120):
        assert pdf_bytes and pdf_bytes.startswith(b"%PDF")
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.pdf")
            out_path = os.path.join(td, "out.pdf")

            with open(in_path, "wb") as f:
                f.write(pdf_bytes)

            cmd = f'gs -sDEVICE=pdfwrite -dCompatibilityLevel={compatibility_level} -dNOPAUSE -dBATCH -dSAFER -dPDFSETTINGS={pdfsettings} -dDetectDuplicateImages=true -dCompressFonts=true -dSubsetFonts=true -sOutputFile="{out_path}" "{in_path}"'

            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Ghostscript failed (code {proc.returncode}).\n"
                    f"STDERR:\n{proc.stderr.decode('utf-8', errors='replace')}"
                )

            with open(out_path, "rb") as f:
                return f.read()

class GeminiWrapper(LLMWrapper):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.response = None
        self.stats = []

    def _send(self, message, pdf_file):
        pdf_file = genai.types.Part.from_bytes(data=pdf_file, mime_type='application/pdf')
        self.response = self.client.models.generate_content(model=self.model_name, contents=[pdf_file, message])
        t_stats = {
            "total": self.response.usage_metadata.total_token_count,
            "prompt_total": self.response.usage_metadata.prompt_token_count,
            "output": self.response.usage_metadata.candidates_token_count
        }
        t_stats["thoughts"] = getattr(self.response.usage_metadata, "thoughts_token_count", None)
        
        try:
            t_stats["prompt_text"] = list(filter(lambda x: str(x.modality) == 'MediaModality.TEXT', self.response.usage_metadata.prompt_tokens_details))[0].token_count
            t_stats["prompt_document"] = list(filter(lambda x: str(x.modality) == 'MediaModality.DOCUMENT', self.response.usage_metadata.prompt_tokens_details))[0].token_count
        except:
            pass
        self.stats.append(t_stats)
        return self.response.text
    
    def _stats(self):
        return self.stats[-1]

class OpenAIWrapper(LLMWrapper):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.response = None
        self.raw_stats = []
        self.stats = []

    def _send(self, message, pdf_file=None):
        content_list = []
        
        base64_string = base64.b64encode(pdf_file).decode("utf-8")
        content_list.append({
            "type": "input_file",
            "filename": "document.pdf", # Generic filename
            "file_data": f"data:application/pdf;base64,{base64_string}"
        })
        content_list.append({
            "type": "input_text",
            "text": message
        })

        # call the API
        self.response = self.client.responses.create(
            model=self.model_name,
            reasoning={"effort": "high"},
            input=[
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        )

        self.raw_stats.append(self.response.usage)
        self.stats.append({
            "total_tokens": self.response.usage.total_tokens,
            "input_tokens": self.response.usage.input_tokens,
            "output_tokens": self.response.usage.output_tokens
        })

        return self.response.output_text

    def _stats(self):
        return self.stats[-1]

class BedrockWrapper(LLMWrapper):
    def __init__(self, model_name):
        region_map = {
            "us.meta.llama3-1-405b-instruct-v1:0": "us-east-2",
            "mistral.mistral-large-3-675b-instruct": "us-east-2"
        }
        if model_name in region_map.keys():
            region = region_map[model_name]
        else:
            region = "us-east-1"
        super().__init__(model_name)

        self.boto_config = botocore.config.Config(read_timeout=300)
        self.client = boto3.client(
            "bedrock-runtime", 
            region_name=region, 
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            config=self.boto_config
        )
        self.response = None
        self.stats = []

    def _send(self, message, pdf_file=None):
        content_block = []

        content_block.append({
            "text": message
        })
        if pdf_file:
            content_block.append({
                "document": {
                    "format": "pdf",
                    "name": "Document1",
                    "source": {
                        "bytes": pdf_file
                    }
                }
            })
        messages = [{
            "role": "user",
            "content": content_block
        }]

        self.response = self.client.converse(modelId=self.model_name, messages=messages)

        t_stats = {
            "total_tokens": self.response['usage']['totalTokens'],
            "input_tokens": self.response['usage']['inputTokens'],
            "output_tokens": self.response['usage']['outputTokens']
        }
        self.stats.append(t_stats)

        # Extract answer text
        # Bedrock returns a list of content blocks; we grab the first text block.
        output_content = self.response["output"]["message"]["content"]
        answer_text = ""
        for block in output_content:
            if "text" in block:
                answer_text += block["text"]
        return answer_text

    def _stats(self):
        return self.stats[-1]

ALL_LLM_MODELS = {
    "gemini-2.5-flash": ("gemini-2.5-flash", GeminiWrapper),
    "gemini-2.0": ("gemini-2.0-flash", GeminiWrapper),
    "gemini-2.5": ("gemini-2.5-pro-exp-03-25", GeminiWrapper),

    "gpt-5-mini": ("gpt-5-mini-2025-08-07", OpenAIWrapper),
    "gpt-5-nano": ("gpt-5-nano", OpenAIWrapper),
    "gpt-4o-mini": ("gpt-4o-mini", OpenAIWrapper),

    # Amazon Nova
    "nova-premier": ("us.amazon.nova-premier-v1:0", BedrockWrapper),
    "nova-pro": ("amazon.nova-pro-v1:0", BedrockWrapper),
    "nova-lite": ("us.amazon.nova-2-lite-v1:0", BedrockWrapper),

    # Claude
    "claude-opus4.5": ("us.anthropic.claude-opus-4-5-20251101-v1:0", BedrockWrapper),
    "claude-haiku4.5": ("us.anthropic.claude-haiku-4-5-20251001-v1:0", BedrockWrapper),

    # DeepSeek
    "deepseek-r1": ("us.deepseek.r1-v1:0", BedrockWrapper),

    # Llama 3.2
    "llama3.2-90b": ("us.meta.llama3-2-90b-instruct-v1:0", BedrockWrapper),

    # Others
    "qwen3-vl-235b": ("qwen.qwen3-vl-235b-a22b", BedrockWrapper)
    # "mistral-large3-675b": ("mistral.mistral-large-3-675b-instruct", BedrockWrapper)
}

def get_wrapper(model_name):
    if model_name not in ALL_LLM_MODELS:
        raise ValueError(f"Model {model_name} is not supported.")
    model_key, wrapper_class = ALL_LLM_MODELS[model_name]
    return wrapper_class(model_key)