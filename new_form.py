from flask import Flask, request, render_template

import func

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def displaySong():
    if request.method == "POST":
        artist = request.form.get("artist")
        song = request.form.get("song")

        result = func.song_recommender(song, artist)

        return render_template("result.html", result=result)

    return render_template("form.html")

if __name__ == '__main__':
    app.run()
