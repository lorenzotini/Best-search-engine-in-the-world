# Flask version of your Streamlit app

from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ---------------- Helper Functions ----------------

def get_mock_results(query, sentiment_filter=None):


    mock_data  = [
        {
            "title": "Tübingen - Eine Stadt voller Geschichte",
            "description": "Tübingen ist eine malerische Stadt in Baden-Württemberg, bekannt für ihre historische Altstadt, die von engen Gassen und gut erhaltenen Fachwerkhäusern geprägt ist. Die Stadt liegt idyllisch am Neckar und bietet zahlreiche Möglichkeiten für Spaziergänge entlang des Flusses oder durch die grünen Parkanlagen. Besonders beliebt ist eine Stocherkahnfahrt auf dem Neckar, bei der man die Stadt aus einer ganz neuen Perspektive erleben kann. Neben der renommierten Universität, die das Stadtbild und das kulturelle Leben maßgeblich prägt, gibt es viele gemütliche Cafés, traditionelle Restaurants und kleine Boutiquen zu entdecken. Im Sommer finden regelmäßig Feste und Märkte statt, die Besucher aus der ganzen Region anziehen. Auch das Umland von Tübingen lädt mit seinen Weinbergen und Wanderwegen zu Ausflügen ein. Insgesamt verbindet Tübingen auf einzigartige Weise Geschichte, Natur und studentisches Leben.",
            "url": "https://www.tuebingen.de",
            "sentiment": "positive",
            "sentiment_score": 95
        },
        {
            "title": "Tübingen Tourismus",
            "description": "Entdecken Sie die Sehenswürdigkeiten und Aktivitäten in Tübingen...",
            "url": "https://www.tuebingen-tourismus.de",
            "sentiment": "neutral",
            "sentiment_score": 60
        },
        {
            "title": "Besuch in Tübingen",
            "description": "Historische Altstadt, moderne Forschungseinrichtungen und mehr...",
            "url": "https://www.tuebingen.de/besuch",
            "sentiment": "negative",
            "sentiment_score": 40
        }
    ]

    if sentiment_filter:
        mock_data = [result for result in mock_data if result['sentiment'] == sentiment_filter]

    return  mock_data

# ---------------- Routes ----------------

@app.route('/', methods=['GET', 'POST'])
def index():
    query = ""
    results = []
    sentiment_filter = ""
    if request.method == 'POST':
        query = request.form.get('query')
        sentiment_filter = request.form.get('sentiment_filter')
        if query:
            if sentiment_filter:
                mock_data = get_mock_results(query, sentiment_filter)
            else:
                mock_data = get_mock_results(query)
            results = mock_data

    return render_template('index.html', query=query, results=results, sentiment_filter=sentiment_filter)




# ---------------- Run App ----------------

if __name__ == '__main__':
    app.run(debug=True)
