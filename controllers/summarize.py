from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.document_loaders import PyPDFLoader

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding Support
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Summarizer for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

# Taking out the warnings
import warnings 
from warnings import simplefilter

from langchain.prompts import PromptTemplate
# "../data/republican.pdf"
def summarize(partyFile):
    # Load variables from .env file
    load_dotenv()
    
    my_openai_api_key = os.getenv('OPENAI_API_KEY')
    llm = OpenAI(temperature=0, openai_api_key=my_openai_api_key)

    # Load the pdf
    pdf_file_path = partyFile

    if not os.path.isfile(pdf_file_path):
        raise ValueError(f"File path {pdf_file_path} is not a valid file")

    loader = PyPDFLoader(pdf_file_path)

    pages = loader.load()

    # Combine pages and replace tabs with spaces
    text = ""

    for page in pages:
        text += page.page_content

    text = text.replace('\t', ' ')

    num_tokens = llm.get_num_tokens(text)
    print (f"The pdf has {num_tokens} tokens in it.")

    # Split text into large chunks, which will be the new documents
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=5000, chunk_overlap=1500)
    docs = text_splitter.create_documents([text])

    num_documents = len(docs)
    print (f"The pdf has been split into {num_documents} documents.")

    # Create 'embeddings', which is a list of 1536-dimensional embeddings ??
    embeddings = OpenAIEmbeddings(openai_api_key=my_openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    # Convert the list of vectors to a numpy array
    vectors_np = np.array(vectors)

    # Create 11 clusters for key issues
    # # (defense, democracy, economy, education, environment, foreign policy, healthcare, immigration, infrastructure, social issues, tech)
    num_clusters = 11 if num_documents >= 11 else num_documents

    # Perform K-means clustering to find optimal path ??
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    print(f"kmeans labels: {kmeans.labels_}")

    # Filter out FutureWarnings
    simplefilter(action='ignore', category=FutureWarning)

    #Perform t-SNE and reduce to 2 dimensions
    perplexity_value = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    reduced_data_tsne = tsne.fit_transform(vectors_np)

    # Find the cloest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):

        # Get the list of disances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argim to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    # Sort so that chunks are processed in order
    selected_indices = sorted(closest_indices)
    print(f"selected indices: {selected_indices}")

    llm4 = ChatOpenAI(temperature=0,
                    openai_api_key=my_openai_api_key,
                    max_tokens=1000,
                    model='gpt-4o-mini'
                    )

    # Create custom prompts to summarize presidential candidate's political party platform
    map_prompt = """
    You will be given a single passage from a presidential candidate's political party platform document. This section will be inclosed in triple backticks(''').
    Your goal is to give a summary of this section so that a reader will have a full understanding of what the presidential candidate's platform is about.
    Your response should be in at least three paragraphs and fully encompass what was said in the passage.

    '''{text}'''
    FULL SUMMARY:
    """

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    # Manual map reduce
    map_chain = load_summarize_chain(llm=llm4,
                                    chain_type="stuff",
                                    prompt=map_prompt_template
                                    )

    # Retrieve docs, which are represented by the top vectors
    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []

    for i, doc in enumerate(selected_docs):
        # Get the summary from the model using the invoke method
        chunk_summary = map_chain.invoke([doc])
        
        # Extract the actual text content from the result if necessary
        if isinstance(chunk_summary, list) and len(chunk_summary) > 0 and isinstance(chunk_summary[0], Document):
            chunk_summary_text = chunk_summary[0].page_content
        else:
            chunk_summary_text = str(chunk_summary)

        # Append the summary text to the list
        summary_list.append(chunk_summary_text.replace('\\n', '\n'))

    summaries = "\n".join(summary_list)

    # Convert summaries back to a document
    summaries_doc = Document(page_content=summaries)

    print(f"Your total summary has {llm.get_num_tokens(summaries_doc.page_content)} tokens.")

    llm4 = ChatOpenAI(temperature=0,
                    openai_api_key=my_openai_api_key,
                    max_tokens=3000,
                    model='gpt-4o-mini',
                    request_timeout=120
                    )

    combine_prompt = """
    You will be given a series of summaries from a presidential candidate's political party platform document.
    The summaries will be encolsed in triple backticks (''')
    The reader should be able to grasp what happened in the book.

    '''{text}'''
    VERBOSE SUMMARY:
    """

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_chain = load_summarize_chain(llm=llm4,
                                        chain_type="stuff",
                                        prompt=combine_prompt_template,
                                        verbose=True
                                        )

    output = reduce_chain.invoke([summaries_doc])

    print(output)
    return output
