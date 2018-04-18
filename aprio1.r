setwd('g:/datasets/SuperStore')
library (arules)

library (arulesViz)


# Load the data set
mydata <- read.csv("sales_ap.csv", header = TRUE  )

dt <- split(mydata$StockCode, mydata$InvoiceNo)


# Convert data to transaction level
dt2 = as(dt,"transactions")
summary(dt2)
inspect(dt2)
transDat <-dt2

itemFrequency(dt2, type = "relative")
itemFrequencyPlot(dt2,topN=10, type="relative")

rules <- apriori(dt2, parameter = list(supp = 0.005, conf = 0.8, minlen = 3))

quality(rules)

# Show the top 5 rules, but only 2 digits



options (digits=2)

inspect (rules[1:5])

rules <- sort (rules, by="confidence", decreasing=TRUE) # 'high-confidence' rules.

redundant <- which (colSums (is.subset (rules, rules)) > 1) # get redundant rules in vector

rules <- rules[-redundant] # remove redundant rules

# Interactive Plot

plot (rules[1:25],method="graph",shading="confidence", gp_labels= gpar(col = "blue", cex=1, fontface="italic")) # feel free to expand and move around the objects in this plot

  plot (rules, measure=c("support", "lift"), shading="confidence")

subrules <- subset(rules, lift>2.5)
subrules

plot(subrules, method="matrix", measure="lift")
plot(subrules, method="matrix", measure="lift", control=list(reorder=TRUE))

plot(subrules, method="matrix3D", measure="lift")
plot(subrules, method="matrix3D", measure="lift", control=list(reorder=TRUE))

## grouped matrix plot
plot(rules, method="grouped")
plot(rules, method="grouped",
     control = list(col = grey.colors(10),
                    gp_labels= gpar(col = "blue", cex=1, fontface="italic")))
## try: sel <- plot(rules, method="grouped", interactive=TRUE)

subrules2 <- sample(rules, 10)
plot(subrules2, method="graph", gp_labels= gpar(col = "blue", cex=1, fontface="italic"))

plot(subrules2, method="paracoord", control=list(reorder=TRUE))

## Doubledecker plot only works for a single rule

oneRule <- sample(rules, 1)
inspect(oneRule)
plot(oneRule, method="doubledecker", data = dt2)

## for itemsets
itemsets <- eclat(dt2, parameter = list(support = 0.02, minlen=2))
plot(itemsets)
plot(itemsets, method="graph")
plot(itemsets, method="paracoord", control=list(alpha=.5, reorder=TRUE))

