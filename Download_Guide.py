import streamlit as st

def show_guide():
    st.title("📥 How to Download the Excel File")

    st.markdown("""
    ## 1️⃣ Go to 👉 [leghe.fantacalcio.it](https://leghe.fantacalcio.it)
    - Log in with your username and password.

    ## 2️⃣ Select your league
    - After logging in, click on the league you want to get data from.

    ## 3️⃣ Navigate to the "standings" section
    - In the bottom left corner, click **FULL RANKING**.

    ## 4️⃣ Download the Excel file
    - On the top of the page, below "Classification", look for a button labeled:
      - 📤 “DOWNLOAD NOW”
    - Click it to download the `.xlsx` file.

    ## 5️⃣ Upload the file in this app
    - Return here and upload the downloaded file using the uploader.
    - The dashboard will process the data automatically. ✅

    ---
    ⚠️ Make sure to download the file **directly from the site** and that it’s in `.xlsx` format.
    """)
