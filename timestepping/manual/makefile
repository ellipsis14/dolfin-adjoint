LATEXFLAGS=-halt-on-error
BIBTEXFLAGS=

TEX = $(wildcard *.tex)
PDF = $(addsuffix .pdf, $(basename $(TEX)))

default: $(PDF)

%.pdf: %.tex bibliography.bib
	pdflatex $(LATEXFLAGS) $<
	bibtex $(BIBTEXFLAGS) $(basename $<) || true
	pdflatex $(LATEXFLAGS) $<
	pdflatex $(LATEXFLAGS) $<
	pdflatex $(LATEXFLAGS) $<
	pdflatex $(LATEXFLAGS) $<

clean:
	rm -f $(PDF)
	rm -f $(addsuffix .aux, $(basename $(TEX)))
	rm -f $(addsuffix .bbl, $(basename $(TEX)))
	rm -f $(addsuffix .blg, $(basename $(TEX)))
	rm -f $(addsuffix .log, $(basename $(TEX)))
	rm -f $(addsuffix .out, $(basename $(TEX)))
	rm -f $(addsuffix .toc, $(basename $(TEX)))
