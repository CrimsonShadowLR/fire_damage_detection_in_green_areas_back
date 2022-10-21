test_dir = .

# run backend server
run:
	uvicorn api.main:app --port 8086 --reload

# create new migrations
migrations:
	python manage.py makemigrations

# load new migrations in the DB
migrate:
	python manage.py migrate

# libraries

install_requirements:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --upgrade pip-tools
	pip-sync requirements.txt

update_requirements:
	pip-compile requirements.in --output-file=requirements.txt

# django interactive shell
shell:
	python manage.py shell