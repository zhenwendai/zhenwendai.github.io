all: blog.html

blog.html:
		pandoc --mathjax --template=GitHub.html5 math_def.md blog.md -o blog.html --filter pandoc-citeproc
