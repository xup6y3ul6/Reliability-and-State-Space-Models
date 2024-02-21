library(haven)
library(readr)

setwd("C:/Users/P70085765/OneDrive/Projects/RILD/dataset")

data <- read_sav("data_1beep_no1st beep_annette.sav")
View(data)
write_tsv(data, "data_1beep_no1st beep_annette.tsv")
