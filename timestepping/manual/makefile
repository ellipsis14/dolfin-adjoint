LATEXFLAGS=-halt-on-error
BIBTEXFLAGS=

TEX = $(wildcard *.tex)
PDF = $(addsuffix .pdf, $(basename $(TEX)))

default: $(PDF)

%.pdf: %.tex bibliography.bib
	latex $(LATEXFLAGS) $<
	bibtex $(BIBTEXFLAGS) $(basename $<) || true
	latex $(LATEXFLAGS) $<
	latex $(LATEXFLAGS) $<
	latex $(LATEXFLAGS) $<
	latex $(LATEXFLAGS) $<
	dvips $(basename $<) | tee -a latex.log; \
	ps2pdf $(basename $<).ps | tee -a latex.log; \

clean:
	rm -f $(PDF)
	rm -f $(addsuffix .aux, $(basename $(TEX)))
	rm -f $(addsuffix .bbl, $(basename $(TEX)))
	rm -f $(addsuffix .blg, $(basename $(TEX)))
	rm -f $(addsuffix .log, $(basename $(TEX)))
	rm -f $(addsuffix .out, $(basename $(TEX)))
	rm -f $(addsuffix .toc, $(basename $(TEX)))
	rm -f $(addsuffix .dvi, $(basename $(TEX)))
	rm -f $(addsuffix .ps, $(basename $(TEX)))
	rm -f latex.log
