init:
	pip install -r requirements.txt

test:
	python -m unittest

clean:
	find . -name \*.pyc -delete
	find . -name \*.p -delete
