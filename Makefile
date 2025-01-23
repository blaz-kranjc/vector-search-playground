.PHONY: clean deepclean

PYTHON = python3.9

clean:
	rm -rf target/

deepclean: clean
	rm -rf app/.venv/
	rm -rf embeddings/.venv/

%/.venv/: %/requirements.txt
	$(PYTHON) -m venv $@
	$@/bin/pip install -r $<

target:
	mkdir -p target

target/%.db: data/%.txt target embeddings/.venv/
	source embeddings/.venv/bin/activate && python embeddings/main.py -o $@ -c Abstract -c Title $<

run: target/1000-publications.db app/.venv/
	source app/.venv/bin/activate && python app/main.py $<
