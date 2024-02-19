from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA
import torch 
import os
import warnings
warnings.filterwarnings("ignore")


class RAG_LLM:
    def __init__(self, pdf_path, model_path, with_hf_api=False, device = 'cuda', hf_api_key = None):
        self.hf_api_key = hf_api_key
        if with_hf_api:
            if hf_api_key:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_api_key
            else:
                raise ValueError("You need to provide a Hugging Face API key")
        self.device = device
        self.pdf_path = pdf_path
        self.model_path = model_path
        self.with_hf_api = with_hf_api
    
    def load_pdf(self, specific_page = None, page_range = None):
        pdf_loader = PyPDFLoader(self.pdf_path)
        if specific_page:
            pdf_loader.specific_pages = specific_page
            pages = pdf_loader.load_and_split()[specific_page]
        elif page_range:
            pdf_loader.page_range = page_range
            pages = pdf_loader.load_and_split()[page_range[0]:page_range[1]]
        return pdf_loader.load()    
    
    def get_chunks(self, pages, chunk_size = 512, chunk_overlap = 50):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(pages)
        return chunks
    
    def create_faiss_embeddings(self, chunks, model_name = "sentence-transformers/all-MiniLM-L6-v2", save_path = None):
        Embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs={'device': self.device})

        store = FAISS.from_texts([str(chunk) for chunk in chunks], Embeddings)
        return store, Embeddings
    
    def get_prompt(self, prompt = None, ):
        if prompt:
            prompt = ChatPromptTemplate.from_template(prompt)
        else:
            prompt = """
                Use the following piece of information to write a code that does tha following:

                Context: {context}
                Question: {question}

                If you can't get the answer just say I don't know :\n
                """
            prompt = ChatPromptTemplate.from_template(prompt)

        return prompt
    
    def get_model(self, max_length = 500, quantization = None):
        if self.with_hf_api:
            llm = HuggingFaceHub(repo_id = self.model_path, model_kwargs= {'max_length': max_length}, huggingfacehub_api_token = self.hf_api_key)
            
        else:
            bnb_config = None
            if quantization:
                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config=bnb_config, device_map=self.device)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_length)
            llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    def retrieval_qa_chain(self, llm, prompt, store):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=store.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt}
                                        )
        return qa_chain
    
    def qa_bot(self, Embeddings, store, prompt, hf):
        embeddings = Embeddings
        db = store
        qa_prompt = prompt
        qa = self.retrieval_qa_chain(hf, qa_prompt, db)
        return qa
    
    def get_response(self, query, Embeddings, store, prompt, hf):
        qa_result = self.qa_bot(Embeddings, store, prompt, hf)
        response = qa_result({'query': query})
        return response