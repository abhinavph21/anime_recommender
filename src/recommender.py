# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from src.prompt_template import get_anime_prompt

# class AnimeRecommender:
#     def __init__(self,retriever,api_key:str,model_name:str):
#         self.llm = ChatGroq(api_key=api_key,model=model_name,temperature=0)
#         self.prompt = get_anime_prompt()

#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm = self.llm,
#             chain_type = "stuff",
#             retriever = retriever,
#             return_source_documents = True,
#             chain_type_kwargs = {"prompt":self.prompt}
#         )

#     def get_recommendation(self,query:str):
#         result = self.qa_chain({"query":query})
#         return result['result']


from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.retriever = retriever
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0,
        )
        self.prompt = get_anime_prompt()

        self.chain = self.prompt | self.llm

    def get_recommendation(self, query: str):
        try:
            docs = self.retriever.invoke(query)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            response = self.chain.invoke({
                "context": context,
                "input": query
            })

            return response.content

        except Exception as e:
            print("ERROR:", repr(e))
            raise
