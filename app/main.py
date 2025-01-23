import argparse
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import sqlite3
import sqlite_vec
from dataclasses import dataclass

def parse_arguments():
  parser = argparse.ArgumentParser(description="Simple web application to search for similar publications.")
  parser.add_argument("db_file", help="Database file containing the publications")
  return parser.parse_args()


def db_connection(file):
  db = sqlite3.connect(file)
  db.enable_load_extension(True)
  sqlite_vec.load(db)
  db.enable_load_extension(False)
  return db


def create_embedding(model, data):
  return model.encode(data)


@dataclass
class PublicationResult:
  doi: str
  title: str
  abstract: str
  date_issued: str
  related_title: str
  distance: float


def query_embedding(db, query, n):
  embedding = create_embedding(model, query)
  query = f"""
    SELECT p.IdentifierDOI, p.Title, p.Abstract, p.DateIssued, p.RelatedTitle, view.distance
    FROM publications p
    JOIN (
      SELECT rowid, distance
      FROM publications_embeddings
      WHERE embedding MATCH ?
      ORDER BY distance
      LIMIT {n}
    ) view ON p."index" = view.rowid
  """
  result = db.execute(query, (sqlite_vec.serialize_float32(embedding),))
  to_result = lambda r: PublicationResult(
    doi = r[0],
    title = r[1],
    abstract = r[2],
    date_issued = r[3],
    related_title = r[4],
    distance = r[5],
  )
  return [to_result(r) for r in result.fetchall()]


app = Flask(__name__)
db_file = None
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route("/status", methods=["GET"])
def get_status():
  return jsonify({
    "message": "OK",
  })


@app.route("/search/<query>", methods=["GET"])
def search(query):
  n = max(min(int(request.args.get("n", 5)), 10), 1)
  db = db_connection(db_file)
  result = jsonify({
    "n": n,
    "query": query,
    "results": query_embedding(db, query, n),
  })
  db.close()
  return result


@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
  return app.send_static_file("index.html")


if __name__ == "__main__":
  args = parse_arguments()
  db_file = args.db_file
  app.run(debug = True, host = "0.0.0.0", port = 8080)
