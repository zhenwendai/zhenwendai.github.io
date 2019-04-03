---
title:  "Gaussian Process in Practice: Scalability and Uncertainty"
author: Zhenwen Dai
date:   2019-03-28
bibliography: ../GPSS2018/scalable_gp.bib
---


## Scalability is a big challenge for Gaussian process

Gaussian process is

GP computational time meta-analysis.

a gp fit on computational time. (what if we have a million data points?)

What about big computers?

What about parallelism?

So what else we can do?

cover different approaches:

1. Sparse GP
2. covariance matrix inversion
3. distributed GPs

## What does uncertainty in Gaussian process?


How to display math:

$$
p(x|y) \int \sum \yM
$$

abc [@Titsias2009]

You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
