library(seqRFLP)
library(bold)
library(data.table)
library(stringr)
library(readr)
library(fingerprint)
library(dplyr)
library(taxize)
library(rfishbase)
library(worms)


#Set working directory
setwd("C:/Users/username/files")

#set.seed(123)

#Get species list from Fishbase 

#By ecoregion

North_Sea <-species_by_ecosystem(ecosystem = "North Sea", server = "fishbase")
Norwegian_Sea <-species_by_ecosystem(ecosystem = "Norwegian Sea", server = "fishbase")
Mediterranean_Sea<-species_by_ecosystem(ecosystem = "Mediterranean Sea", server = "fishbase")
Guinea_Current<-species_by_ecosystem(ecosystem = "Guinea Current", server = "fishbase")
Barents_Sea <- species_by_ecosystem(ecosystem = "Barents Sea", server = "fishbase")
Baltic_Sea <-species_by_ecosystem(ecosystem = "Baltic Sea", server = "fishbase")
Benguela_Current <-species_by_ecosystem(ecosystem = "Benguela Current", server = "fishbase")
Canary_Current <-species_by_ecosystem(ecosystem = "Canary Current", server = "fishbase")
Celtic_Biscay_Shelf <-species_by_ecosystem(ecosystem = "Celtic-Biscay Shelf", server = "fishbase")
Faroe_Plateau <-species_by_ecosystem(ecosystem = "Faroe Plateau", server = "fishbase")
Greenland_Sea <-species_by_ecosystem(ecosystem = "Greenland Sea", server = "fishbase")
Iberian_Coastal <-species_by_ecosystem(ecosystem = "Iberian Coastal", server = "fishbase")
Iceland_Shelf_and_Sea <-species_by_ecosystem(ecosystem = "Iceland Shelf and Sea", server = "fishbase")
Namib <-species_by_ecosystem(ecosystem = "Namib", server = "fishbase")
Gulf_of_Guinea <-species_by_ecosystem(ecosystem = "Gulf of Guinea", server = "fishbase")

df <- list(data.frame(North_Sea), data.frame(Norwegian_Sea), data.frame(Mediterranean_Sea), data.frame(Guinea_Current),
           data.frame(Barents_Sea),data.frame(Baltic_Sea), data.frame(Benguela_Current), data.frame(Canary_Current),
           data.frame(Celtic_Biscay_Shelf), data.frame(Faroe_Plateau), data.frame(Greenland_Sea),data.frame(Iberian_Coastal),
           data.frame(Iceland_Shelf_and_Sea), data.frame(Namib), data.frame(Gulf_of_Guinea))

combined_df <- bind_rows(df)

combined_df2<-combined_df[combined_df$Status != "error" & combined_df$Status !="misidentification" & combined_df$Status !="questionable",]

species_list_regions<-data.frame(combined_df2$Species)
names(species_list_regions)<-"species"

#By country
countries<-country()

countries_list<-c("Cape Verde","Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast","Morocco",
                  "Mauritania","Senegal","Togo","Benin","Nigeria","Cameroon","Eq Guinea","Gabon",
                  "Congo","Congo Dem Rep","Sierra Leone","Angola","Namibia","Portugal","Spain","UK","Ireland","France","Netherlands",
                  "Belgium","Denmark","Germany","Sweden","Finland","Poland","Norway","Lithuania","Estonia","Latvia",
                  "Algeria","Tunisia","Malta","Libya","Egypt","Israel","Cyprus","Albania","Croatia","Syria")

countries2<-countries[countries$country %in% countries_list,]
countries3<-countries2[countries2$Status != "error" & countries2$Status !="misidentification" & countries2$Status !="questionable",]

species<-species_names(countries3$SpecCode)

species_list_countries<-data.frame(species$Species)
names(species_list_countries)<-"species"

#Total list
species_list_total<-unique(rbind(species_list_countries,species_list_regions))
species_list_total$word_count <- str_count(species_list_total$species, "\\S+")

#Filter only marine species
species_bold<-wormsbynames(species_list_total$species,marine_only=FALSE, ids=TRUE)
species_bold2<-subset(species_bold, species_bold$isMarine==1)
species_bold3<-species_bold2%>%
  mutate(new_name = ifelse(status=="unaccepted" & !is.na(status),valid_name,
                           ifelse(status=="accepted" & !is.na(status),name,name)))
species_bold4<-species_bold3[c("new_name","family","order","class")]

names(species_bold4)[1]<-"Species"

#Retrieve data from BOLD for Eastern Atlantic marine fish using species list
file_spb<-"https://raw.githubusercontent.com/tadeu95/BAGS/master/species_per_bin.txt"
spb<-fread(file_spb)
file_bps<-"https://raw.githubusercontent.com/tadeu95/BAGS/master/bin_per_species.txt"
bps<-fread(file_bps)

species <- unique(species_bold4$Species)
x <- length(species)
y <- ceiling(x / 300) + 1
taxon_total <- data.frame()
i <- 1
while (i < y) {
  ini <- 1 + (300 * (i - 1))
  fin <- min(300 * i, x)
  tryCatch({
    tmp <- bold_seqspec(taxon = species[ini:fin], response = TRUE)
    tt <- paste0(rawToChar(tmp$content, multiple = TRUE), collapse = "")
    Encoding(tt) <- "UTF-8"
    taxa <- utils::read.delim(text = tt, header = TRUE, sep = "\t", stringsAsFactors = FALSE, quote = "")
    taxon_total <- rbind(taxon_total, taxa)
  }, error = function(e) {
    cat("Error in iteration", i, ": ", conditionMessage(e), "\n")
  })
  
  i <- i + 1
}
taxon<-taxon_total
taxon2 <- taxon[taxon$species_name != "" & !is.na(taxon$species_name), ]
taxon2 <- taxon2[taxon2$bin_uri != "" & !is.na(taxon2$bin_uri), ]

#Retrieve records assigned to all the previously mined BINs
bins<-unique(taxon2$bin_uri)
x<-length(bins)
taxon_total = data.frame()
y <- ceiling(x / 300) + 1
i <- 1
while (i < y){
  ini<-1+(300*(i-1))
  fin <- min(300 * i, x)
  tmp <- bold_seqspec(bin=bins[ini:fin], response = TRUE)
  tt <- paste0(rawToChar(tmp$content, multiple = TRUE), collapse = "")
  Encoding(tt) <- "UTF-8"
  taxa <- utils::read.delim(text = tt, header = TRUE, sep = "\t", stringsAsFactors = FALSE, quote = "")
  taxon_total <- rbind(taxon_total,taxa)
  i = i+1
}
remaining_bins<-taxon_total

remaining_bins2<-remaining_bins[!(remaining_bins$processid %in% taxon2$processid),]
remaining_bins2<-remaining_bins2[remaining_bins2$species_name!=""|is.na(remaining_bins2$species_name),]
remaining_bins2<-remaining_bins2[!(remaining_bins2$bin_uri == "" | is.na(remaining_bins2$bin_uri)), ]

taxon2<-rbind(taxon2,remaining_bins2)

#Preprocess data and BAGS-like pipeline for grade assignment
taxon2<-left_join(taxon2,bps,by="species_name")
taxon2$bin_uri.y=NULL
names(taxon2)[names(taxon2) == "bin_uri.x"] <- "bin_uri"
taxon2<-left_join(taxon2,spb,by="bin_uri")
taxon2$species_name.y=NULL
names(taxon2)[names(taxon2) == "species_name.x"] <- "species_name"
taxon3<-taxon2[taxon2$markercode=="COI-5P",]
taxon3$nucleotides=gsub("[^ATGCNRYSWKMBDHV]+", "", taxon3$nucleotides)
taxon3$nucleotides=gsub("-","",taxon3$nucleotides)
taxon8<-data.frame(taxon3$species_name,taxon3$bin_uri,taxon3$nucleotides,taxon3$country,taxon3$family_name,taxon3$order_name,taxon3$class_name,taxon3$sampleid,taxon3$processid,taxon3$species_per_bin,taxon3$bin_per_species, taxon3$lat,taxon3$lon,taxon3$identification_provided_by,taxon3$institution_storing,taxon3$trace_ids,taxon3$image_ids)
names(taxon8)<-c("species_name","BIN","sequence","country","family","order","class","sampleid","processid","species_per_bin","bin_per_species","latitude","longitude","identified_by","institution_storing","trace_ids","image_ids")
taxon8$grade=NA
names(taxon8)<-c("species","BIN","sequence","country","family","order","class","sampleid","processid","species_per_bin","bin_per_species","latitude","longitude","identified_by","institution_storing","trace_ids","image_ids","grade")
taxon19<-taxon8 %>%
  mutate(grade = ifelse(species_per_bin>1,"E",
                        ifelse(bin_per_species>1 & species_per_bin==1,"C",
                               ifelse(bin_per_species==1 & species_per_bin==1,"AB","needs_update"))))
dominant_grade <- "E"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "C"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "AB"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
taxon19$contains_dominant=NULL
taxon19$base_number=str_count(taxon19$sequence, pattern="[A-Z]")
np=(str_count(taxon19$sequence, "N")/str_count(taxon19$sequence, "[A-Z]"))*100
taxon19$n_percent=np
num_species=table(taxon19$species)
num_species=as.data.frame(num_species)
names(num_species)=c("species","frequency_species")
taxon19<-inner_join(taxon19,num_species)
taxon19<-taxon19 %>%
  mutate(grade = ifelse(grade=="E","E",
                        ifelse(frequency_species<3,"D",
                               ifelse(grade=="C","C",
                                      ifelse(grade=="AB" & frequency_species<11,"B",
                                             ifelse(grade=="AB" & frequency_species>10,"A","needs_update"))))))
dominant_grade <- "E"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "D"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "C"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "B"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
dominant_grade <- "A"
dt <- as.data.table(taxon19)
dt[, contains_dominant := any(grade == dominant_grade), by=species]
dt[contains_dominant == TRUE, grade := dominant_grade]
taxon19 <- setDF(dt)
taxon19$contains_dominant=NULL
taxon19 <- taxon19 %>%
  mutate(grade = ifelse(is.na(grade), "needs_update", grade))
taxon19 <- taxon19 %>%
  mutate(country = ifelse(country=="", "no_country", country))
taxon19 <- taxon19 %>%
  mutate(institution_storing = ifelse(institution_storing=="", "no_institution", institution_storing))
taxon19 <- taxon19 %>%
  mutate(identified_by = ifelse(identified_by=="", "no_identifier", identified_by))
taxon19<-taxon19[order(taxon19$species),]

#write_tsv(taxon19,"taxon19.tsv")
#taxon19<-read.delim("taxon19.tsv")

#Retain only Grade E species
grade_e<-taxon19[taxon19$grade=="E",]

#Count sequences in each BIN
otu_counts <- table(grade_e$BIN)

#grade_e$total_records_in_bin <- otu_counts[grade_e$BIN]
grade_e$total_records_in_bin <- ave(seq_along(grade_e$BIN), grade_e$BIN, FUN = length)

#Calculate the number and percentage of sequences in a BIN, belonging to a particular species
count_species_otu <- grade_e %>%
  group_by(species, BIN) %>%
  dplyr::summarise(species_records_in_bin = n()) %>%
  ungroup()

grade_e2<-left_join(grade_e,count_species_otu,by=c("species","BIN"))

grade_e2$percent_of_bin_records_belonging_to_species<-(grade_e2$species_records_in_bin/ grade_e2$total_records_in_bin)*100

#Count the number of different taxonomic identifiers
grade_e2$species_bin<-paste(grade_e2$species,grade_e2$BIN,sep="|")

identifiers_number <- grade_e2 %>%
  group_by(species_bin) %>%
  dplyr::summarise(unique_identifiers = n_distinct(identified_by[identified_by != "no_identifier"]), .groups = 'drop')

grade_e3<-left_join(grade_e2,identifiers_number,by="species_bin")

#Check for synonyms for each species in the data set
sinonimos<-taxize::synonyms(unique(grade_e3$species),db="worms")

result_df <- data.frame(Name = character(), Synonyms = character(), stringsAsFactors = FALSE)
for (i in seq_along(sinonimos)) {
  species_name <- names(sinonimos)[i]
    if (is.data.frame(sinonimos[[i]]) && nrow(sinonimos[[i]]) > 0) {
    if ("scientificname" %in% colnames(sinonimos[[i]])) {
      synonyms <- paste(sinonimos[[i]]$scientificname, collapse = ", ")
      result_df <- rbind(result_df, data.frame(Name = species_name, Synonyms = synonyms, stringsAsFactors = FALSE))
    }
  }
}
names(result_df)<-c("species","synonyms")

grade_e4 <- left_join(grade_e3, result_df, by = c("species" = "species")) %>%
  mutate(synonyms = ifelse(is.na(synonyms), "no_synonyms", synonyms))

grade_e5<-grade_e4[,c("processid","species","BIN","species_bin","species_per_bin","bin_per_species",
                      "percent_of_bin_records_belonging_to_species","identified_by","unique_identifiers","institution_storing","synonyms",
                      "frequency_species")]

grade_e5$synonyms_list <- lapply(strsplit(as.character(grade_e5$synonyms), ","), trimws)

unique_species <- unique(unlist(grade_e5$species))

synonyms<-result_df

filter_synonyms <- function(synonyms, valid_species) {
  valid_synonyms <- intersect(synonyms, valid_species)
  if (length(valid_synonyms) > 0) {
    return(valid_synonyms)
  } else {
    return("no_synonyms")
  }
}

grade_e5$filtered_synonyms <- mapply(filter_synonyms, grade_e5$synonyms_list, list(unique_species))

grade_e5$filtered_synonyms_string <- sapply(grade_e5$filtered_synonyms, function(x) ifelse(x == "no_synonyms", x, paste(x, collapse = ",")))
grade_e5$synonyms_list<-NULL
grade_e5$synonyms<-NULL
grade_e5$filtered_synonyms<-NULL

#Round the percentage column
grade_e5$percent_of_bin_records_belonging_to_species <- round(grade_e5$percent_of_bin_records_belonging_to_species, 1)

#Order rows randomly if needed
grade_e6 <- grade_e5[sample(nrow(grade_e5)), ]

#Count total sequences in each cluster
#otu_counts <- table(grade_e6$BIN)

#Add a new column to the output data frame with the count of sequences in each BIN
grade_e6$total_records_in_bin <- ave(seq_along(grade_e6$BIN), grade_e6$BIN, FUN = length)

count_species_otu <- grade_e6 %>%
  group_by(species, BIN) %>%
  dplyr::summarise(species_records_in_bin = n()) %>%
  ungroup()

grade_e6<-left_join(grade_e6,count_species_otu,by=c("species","BIN"))

#Add genus column if needed
grade_e6$genus<-str_extract(grade_e6$species, '[A-Za-z]+')

#Add yes or no ambiguous column
grade_e7 <- grade_e6 %>%
  mutate(ambiguous_name = ifelse(grepl("\\.|\\d", species) | str_count(species, "\\S+") > 2, "yes", "no"))

#Use dplyr to update the synonym column
grade_e8 <- grade_e7 %>%
  mutate(synonym = ifelse(filtered_synonyms_string == "no_synonyms", species[match(species, filtered_synonyms_string)], filtered_synonyms_string))

grade_e8$synonym <- ifelse(is.na(grade_e8$synonym), "no_synonyms", grade_e8$synonym)

grade_e8<-grade_e8 %>%
  mutate(synonym = ifelse(species == synonym, "no_synonyms", synonym))

#Count number of unique institution storings
grade_e9<-grade_e8 %>%
  mutate(institution_storing=ifelse(institution_storing=="Mined from GenBank, NCBI"| institution_storing=="*unvouchered",
                                    "no_institution",institution_storing))

institution_storing_df <- grade_e9 %>%
  group_by(species_bin) %>%
  dplyr::summarise(unique_institutions = n_distinct(institution_storing[institution_storing != "no_institution"]), .groups = 'drop')

grade_e10<-left_join(grade_e9,institution_storing_df,by="species_bin")

grade_e10$filtered_synonyms_string=NULL

grade_e10$species_bin=NULL

#Determine if species has a synonym in the respective BIN
grade_e10 <- grade_e10 %>%
  group_by(BIN) %>%
  group_modify(~ {
    current_species <- unique(.x$species)
    .x %>%
      mutate(
        ingroup_synonym = if_else(
          synonym != "no_synonyms" & 
            synonym != species & 
            synonym %in% current_species,
          "yes",
          "no"
        )
      )
  }) %>%
  ungroup()


#Keep only discordant BINs
grade_e10<-grade_e10[grade_e10$species_per_bin>1,]

grade_e10 <- grade_e10 %>%
  mutate(p = percent_of_bin_records_belonging_to_species / 100)

species_bin <- grade_e10 %>%
  distinct(BIN, species, p)

entropy_per_bin  <- species_bin %>%
  group_by(BIN) %>%
  dplyr::summarise(
    shannon_entropy = -sum(p * log(p)),
    .groups = "drop"
  )

grade_e10 <- left_join(grade_e10, entropy_per_bin, by = "BIN")

grade_e10$p<-NULL

grade_e10$shannon_entropy <- round(grade_e10$shannon_entropy, 4)

#Check orioirtuib for each genus
genus_bin_counts <- grade_e10 %>%
  group_by(BIN, genus) %>%
  dplyr::summarise(genus_count = n(), .groups = "drop")

bin_counts <- grade_e10 %>%
  group_by(BIN) %>%
  dplyr::summarise(total_bin_count = n(), .groups = "drop")

genus_prop_df <- left_join(genus_bin_counts, bin_counts, by = "BIN") %>%
  dplyr::mutate(genus_prop_in_bin = genus_count / total_bin_count)

grade_e10 <- left_join(grade_e10, genus_prop_df, by = c("BIN", "genus"))

grade_e10$genus_prop_in_bin<-grade_e10$genus_prop_in_bin*100

grade_e10$total_bin_count<-NULL
grade_e10$genus_count<-NULL
grade_e10$genus<-NULL

#write_tsv(grade_e10,"grade_e10.tsv")

#Prepare file for manual inspection and labelling (in MS excel for example)
df_ordered <- grade_e10[order(grade_e10$BIN, grade_e10$species), ]

df_ordered<-df_ordered[df_ordered$species_per_bin>1,]

df_ordered2<-df_ordered[,c("processid","species","BIN","species_per_bin","bin_per_species","percent_of_bin_records_belonging_to_species",
                           "identified_by","unique_identifiers","institution_storing","unique_institutions","frequency_species","total_records_in_bin","species_records_in_bin",
                           "ambiguous_name","synonym","ingroup_synonym","shannon_entropy","genus_prop_in_bin")]


#grade_e10$identified_by<-NULL
#grade_e10$institution_storing<-NULL


#Data set for manual labelling of each record as supported or inconclusive (add a column named 'label' in excel)
write_tsv(df_ordered2,"df_ordered2.tsv")

#After labelling each record as supported or inconclusive in excel, read back the data set to convert the features for deep learning
grade_e11<-read.table("clipboard",sep="\t",header=TRUE,dec=".",quote = "")

grade_e11$species <- NULL
grade_e11$synonym <- NULL
grade_e11$identified_by<-NULL
grade_e11$institution_storing<-NULL

#Transform features for ML
grade_e11$BIN_nn <- as.numeric(factor(grade_e11$BIN, levels = unique(grade_e11$BIN)))

grade_e11$species_per_bin_nn<-as.numeric(grade_e11$species_per_bin)

grade_e11$bin_per_species_nn<-as.numeric(grade_e11$bin_per_species)

grade_e11$percent_of_bin_records_belonging_to_species_nn<-grade_e11$percent_of_bin_records_belonging_to_species / 100

grade_e11$genus_prop_in_bin_nn<-grade_e11$genus_prop_in_bin / 100

grade_e11$frequency_species_nn<-as.numeric(grade_e11$frequency_species)

grade_e11$total_records_in_bin_nn<-as.numeric(grade_e11$total_records_in_bin)

grade_e11$species_records_in_bin_nn<-as.numeric(grade_e11$species_records_in_bin)

grade_e11$ambiguous_name_nn <- as.numeric(grade_e11$ambiguous_name == "yes")

grade_e11$ingroup_synonym_nn<-as.numeric(grade_e11$ingroup_synonym == "yes")

grade_e11$shannon_entropy_nn<-grade_e11$shannon_entropy

grade_e11$unique_identifiers_nn<-as.numeric(grade_e11$unique_identifiers)

grade_e11$unique_institutions_nn<-as.numeric(grade_e11$unique_institutions)

write_tsv(grade_e11,"grade_e11.tsv")

grade_e12<-grade_e11[,c("processid","BIN_nn","species_per_bin_nn","bin_per_species_nn",
                        "percent_of_bin_records_belonging_to_species_nn","genus_prop_in_bin_nn",
                        "frequency_species_nn","total_records_in_bin_nn","species_records_in_bin_nn"
                        ,"ambiguous_name_nn","ingroup_synonym_nn","shannon_entropy_nn","unique_identifiers_nn","unique_institutions_nn","label")]


grade_e12$ground_truth_label <-as.numeric(grade_e12$label == "supported")

write_tsv(grade_e12,"grade_e12.tsv")

#Check for NAs in the dataset
sum(is.na(grade_e12))

grade_e12$label<-NULL

#Save tsv file to be used for model training and testing
write_tsv(grade_e12,"labelled_for_nn_ready.tsv")


