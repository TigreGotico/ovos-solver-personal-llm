# inspired by: https://github.com/rushic24/langchain-remember-me-llm/
# MIT license
import torch
from json_database import JsonStorageXDG
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from llama_index import Document
from llama_index import LLMPredictor, ServiceContext
from llama_index import LangchainEmbedding, GPTVectorStoreIndex as GPTSimpleVectorIndex
from ovos_plugin_manager.templates.solvers import QuestionSolver
from transformers import pipeline


class UserInfo:
    db = JsonStorageXDG("personalLLM")
    db.setdefault("data", [])

    @classmethod
    def remember(cls, fact):
        cls.db["data"].append(fact)
        cls.db.store()


class PersonalLLMSolver(QuestionSolver):
    enable_tx = True
    priority = 80

    def __init__(self, config=None):
        config = config or {}
        config["lang"] = "en"  # only english supported (not really, depends on model... TODO)
        super().__init__(config)

        # a class inside a class :O
        class PersonalUserLLM(LLM):
            model_name = config.get("model") or "google/flan-t5-small"
            pipeline = pipeline("text2text-generation", model=model_name, device=0,
                                model_kwargs={"torch_dtype": torch.bfloat16})
            initial_prompt = config.get("initial_prompt") or \
                             'You are a highly intelligent question answering A.I. based on the information provided by the user. ' \
                             'If the answer cannot be found in the user provided information, write "I could not find an answer."'

            @classmethod
            def get_engine(cls):
                llm_predictor = LLMPredictor(llm=cls())
                hfemb = HuggingFaceEmbeddings()
                embed_model = LangchainEmbedding(hfemb)
                documents = [Document(t) for t in UserInfo.db["data"]]
                service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
                index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
                return index.as_query_engine()

            def _call(self, prompt, stop=None):
                text = f"{self.initial_prompt}\n\n{prompt} {stop}" if stop is not None else f"{self.initial_prompt}\n\n{prompt}"
                return self.pipeline(text, max_length=9999)[0]["generated_text"]

            @property
            def _identifying_params(self):
                return {"name_of_model": self.model_name}

            @property
            def _llm_type(self):
                return "custom"

        self.llm = PersonalUserLLM.get_engine()

    # officially exported Solver methods
    def get_spoken_answer(self, query, context=None):
        return self.llm.query(query).response
