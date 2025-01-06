.ONESHELL:

share-env:
	conda env export --from-history > environment.yml
	pip list --format=freeze > requirements.txt

clean:
	rm -Ir data

pdf:
	cd paper; pdflatex paper.tex; bibtex paper; pdflatex paper.tex; pdflatex paper.tex
	make clean-pdf

clean-pdf:
	@echo "Cleaning up..."
	cd paper; rm *.aux *.out *.log *.bbl *.blg
	@echo "Cleanup complete."
