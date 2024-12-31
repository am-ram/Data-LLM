import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq

llm = ChatGroq( model_name="llama3-8b-8192", api_key=st.secrets["GROQ_API_KEY"])

# def main():
#     uploaded_file = st.file_uploader('Upload a CSV File...',type=['csv'])
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
#         st.write(data.head(5))
#         df = SmartDataframe(data, config={"llm":llm})
#         prompt = st.text_area("Enter a question to get Insights on your Data....")
#         if st.button("Generate"):
#             if prompt:
#                 with st.spinner("Thinking...."):
#                     st.write(df.chat(prompt))
#             else:
#                 st.warning("Prompt can't be blank!!")

# if __name__ == "__main__":
#     main()
st.title("AI-Based CSV Analysis")
def chat_with_csv(df,prompt):
    # llm = OpenAI(api_token=openai_api_key)
    # pandas_ai = PandasAI(llm)
    sdf = SmartDataframe(df, config={"llm":llm})
    result = sdf.chat(prompt)
    print(result)
    return result

# st.set_page_config(layout='wide')


input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:

        col1, col2 = st.columns([1,1])

        with col1:
            st.info("CSV Uploaded Successfully")
            data = pd.read_csv(input_csv)
            st.dataframe(data, use_container_width=True)

        with col2:

            st.info("Chat Below")
            
            input_text = st.text_area("Enter your query")

            if input_text is not None:
                if st.button("Chat with CSV"):
                    st.info("Your Query: "+input_text)
                    result = chat_with_csv(data, input_text)
                    st.success(result)
