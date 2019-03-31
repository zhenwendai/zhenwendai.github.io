all: blog slides

slides: slides_pdf

blog:
		pandoc --mathjax --template=GitHub.html5 math_def.md blog.md -o blog.html --filter pandoc-citeproc

slides_pdf:
		pandoc --mathjax --filter pandoc-citeproc -t beamer -V theme:Boadilla -V fonttheme:professionalfonts math_def.md slides.md -o slides.pdf

slides_html:
		pandoc -s --mathjax --filter pandoc-citeproc -t revealjs -V revealjs-url=https://revealjs.com -V theme=beige math_def.md slides.md -o slides.html
