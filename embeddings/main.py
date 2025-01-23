import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import sqlite_vec

def parse_arguments():
  parser = argparse.ArgumentParser(description="Generate embeddings from a TSV file.")
  parser.add_argument("csv_file", help="Path to the TSV file to extract embeddings from.")
  parser.add_argument("-o", "--output", help="Path to the output database file.", default="output.sqlite")
  parser.add_argument("-c", "--column", help="Column to use for embeddings generation.", action="append", required=True)
  return parser.parse_args()


def db_connection(file):
  db = sqlite3.connect(file)
  db.enable_load_extension(True)
  sqlite_vec.load(db)
  db.enable_load_extension(False)
  return db


def read_tsv(file_path):
  return pd.read_csv(file_path, sep="\t")


def assert_columns(data, columns):
  missing_columns = [col for col in columns if col not in data.columns]
  if missing_columns:
      raise ValueError(f"CSV file is missing the following columns: {', '.join(missing_columns)}")


def create_embedding(model, data):
  return model.encode(data).astype(np.float32)


def create_embeddings(data, columns):
  assert_columns(data, columns)
  model = SentenceTransformer('all-MiniLM-L6-v2')

  return data[columns].apply(lambda row: create_embedding(model, " ".join(row.values.astype(str))), axis=1)


def create_embedding_table(db):
  db.execute("DROP TABLE IF EXISTS publications_embeddings")
  db.execute("CREATE VIRTUAL TABLE publications_embeddings USING vec0(embedding float[384])")


def store_data_with_embeddings(db, data, embeddings):
  data.to_sql("publications", db, if_exists="replace", index=True)
  create_embedding_table(db)
  db.cursor().executemany(
    "INSERT INTO publications_embeddings (rowid, embedding) VALUES (?, ?)",
    enumerate(embeddings.to_list())
  )


if __name__ == "__main__":
  args = parse_arguments()
  data = read_tsv(args.csv_file)
  embeddings = create_embeddings(data, args.column)
  connection = db_connection(args.output)
  store_data_with_embeddings(connection, data, embeddings)
  connection.commit()
  connection.close()
