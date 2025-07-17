import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import pandas as pd
import numpy as np

def show_result(result):
    """
    Displays a search result in a card format.
    """

    # --- 1. Suchergebnis Box ---

    with st.container():
        st.markdown(f"### {result['title']}")
        
        # Metadatenzeile
        st.caption("📅 Veröffentlicht am: 12. Juli 2025   📰 Quelle: Wikipedia")

        # Zusammenfassung
        st.write(f"{result['description']}\n[More]({result['url']})")

        st.write(f"**Subjektivität:** {result['subjectivity']}")

    # --- 2. Sentiment Analyse Karten ---

    # Example value: model confidence
    confidence = 0.95  # 95%

    # Plot
    fig, ax = plt.subplots(figsize=(5, 0.1))  # small height

    # One bar
    ax.barh([''], [confidence], height=0.1, color='#5ecf9c')  # green bar

    # Remove x-axis, y-axis, borders
    ax.set_xlim(0, 1)
    ax.axis('off')  # remove everything

    fig.tight_layout(pad=0, h_pad=0, w_pad=0)  # tight layout

    # Show in Streamlit
    st.pyplot(fig)



def search(query):
    # some function which returns a dictionary with search results
    # For demonstration, we will just return a mock result
    results = [{
        "title": "Tübingen - Eine Stadt voller Geschichte",
        "description": "Tübingen ist eine malerische Stadt in Baden-Württemberg, bekannt für ihre alte Universität und die wunderschöne Altstadt.",
        "url": "https://www.tuebingen.de",
        "subjectivity": "0.854",
        "sentiment": "POSITIVE",
        "sentiment_score": "0.95"
    },
    {
        "title": "Tübingen Tourismus",
        "description": "Entdecken Sie die Sehenswürdigkeiten und Aktivitäten in Tübingen. Von historischen Gebäuden bis zu modernen Attraktionen.",
        "url": "https://www.tuebingen-tourismus.de",
        "subjectivity": 0.854,
        "sentiment": "Negative",
        "sentiment_score": 0.95
    },    
    {
        "title": "Tübingen Tourismus",
        "description": "Entdecken Sie die Sehenswürdigkeiten und Aktivitäten in Tübingen. Von historischen Gebäuden bis zu modernen Attraktionen.",
        "url": "https://www.tuebingen-tourismus.de",
        "subjectivity": 0.854,
        "sentiment": "Neutral",
        "sentiment_score": 0.95
    },
    {
        "title": "Tübingen Tourismus",
        "description": "Entdecken Sie die Sehenswürdigkeiten und Aktivitäten in Tübingen. Von historischen Gebäuden bis zu modernen Attraktionen.",
        "url": "https://www.tuebingen-tourismus.de",
        "subjectivity": 0.854,
        "sentiment": "POSITIVE",
        "sentiment_score": 0.95
    }
    ]

    if results:
        for result in results:
            show_result(result)
    else:
        st.warning("No results found for your query.")


def main():

    # Configure the page
    st.set_page_config(page_title="Tübingen Search", layout="centered")

    # Load the CSS from file
    with open("interface_style.css") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

    st.markdown("## 🔍 Tübingen Search")

    # Input field with placeholder, styled like a search bar
    query = st.text_input(
        "", 
        placeholder="Suche nach Informationen über Tübingen …", 
        label_visibility="collapsed"
    )
    

    # Search button
    if st.button("🔎 Suchen") or query.strip():
        if query.strip() != "":
            st.success(f"Searchresults for: **{query}**")
            search(query.strip())
        else:
            st.warning("Please enter a search term.")


main()