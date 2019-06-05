library(shiny)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(rattle)
library(Metrics)
library(randomForest)
library(xgboost)

Data <- read.csv("Data.csv")
ClData <- select(Data, - c("X", "Path", "Dists"))
str(ClData)

# Set seed and create assignment
set.seed(1)
assign <- sample(1:3, size = nrow(ClData), prob = c(0.7, 0.15, 0.15), replace = TRUE)

# Create a train and tests from the original data frame 
Data_train <- ClData[assign == 1, ]    # subset ETA_JAM to training indices only
Data_valid <- ClData[assign == 2, ]  # subset ETA_JAM to validation set only
Data_test <- ClData[assign == 3, ]   # subset ETA_JAM to test indices only

ui <- fluidPage(
  # App title ----
  titlePanel("Tree - Based Models And ETA Jam Prediction"),
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    # Sidebar panel for inputs ----
    sidebarPanel(

      # Input: Choose dataset ----
      selectInput("dataset", "Choose a Dataset:",
                  choices = c("Clear Data","Data")),
      
      # Input: Choose TARGET ----
      selectInput("Tar", "Target Variable:",
                  choices = names(Data),
                  selected = "ETA_JAM"),
      # Input: Choose VAR ----
      selectInput("Var", "Explanatory Variable(s):",
                  choices = names(Data),
                  multiple = T,
                  selected = "Distance"),
      hr(),
      h5('Scatterplot'),
      checkboxInput("plot",
                    label = "Modify Scatterplot",
                    value = FALSE),
      conditionalPanel(condition = 'input.plot == true',
                       selectInput("x",
                                   label    = "X-AXIS",
                                   selected = "ETA_JAM",
                                   choices  = names(Data),
                                   selectize = FALSE),
                       selectInput("y",
                                   label    = "Y-AXIS",
                                   selected = "Distance",
                                   choices = names(Data),
                                   selectize = FALSE)),
      hr(),
      radioButtons("Model", "Choose The Model",
                   choices = c("Desicion Tree","Random Forest","XGBoosting"),
                   selected = "XGBoosting"),
      br(),
      
      #selectInput("Mod", "Choose The Model:",
                  #choices = c("BEST_DT","BEST_RF","BEST_XGB")),
      
      h4("Let's Tune Them"),
      h5('Desicion Tree Model Hyperparameters'),
      checkboxInput("Hyp",
                    label = "Model Hyperparameters",
                    value = FALSE),
      conditionalPanel(condition = 'input.Hyp == true',
                       sliderInput("minsplit",
                                   label    = "Minsplit",
                                   min = 1, max = 30,
                                   value = 15,
                                   step = 1,
                                   p('The minimum number of observations that must exist in a node in order for a split to be attempted')),
                       sliderInput("maxdepth",
                                   label = "Maxdepth",
                                   min = 1, max = 30,
                                   value = 10,
                                   step = 1,
                                   p("Set the maximum depth of any node of the final tree, with the root node counted as depth 0")),
                       sliderInput("cp",
                                   label = "Complexity Parameter(CP)",
                                   min = 0, max = 1,
                                   value = 0,
                                   step = 0.01,
                                   p("The main role of this parameter is to save computing time by pruning off splits that are obviously not worthwhile"))
      ),
      
      h5('Random Forest Model Hyperparameters'),
      checkboxInput("Hp",
                    label = "Model Hyperparameters",
                    value = FALSE),
      conditionalPanel(condition = 'input.Hp == true',
                       sliderInput("mtry",
                                   label = "Mtry",
                                   min = 1, max = 30,
                                   value = 5,
                                   step = 1,
                                   helpText('Number of variables randomly sampled as candidates at each split')),
                       sliderInput("nodesize",
                                   label = "Nodesize",
                                   min = 1, max = 30,
                                   value = 7,
                                   step = 1,
                                   helpText("Minimum size of terminal nodes")),
                       sliderInput("ntree",
                                   label = "Ntree",
                                   min = 1, max = 500,
                                   value = 35,
                                   step = 1, 
                                   helpText("Number of trees to grow"))
      ),
      
      h5('XGBoosting Model Hyperparameters'),
      checkboxInput("H",
                    label = "Model Hyperparameters",
                    value = FALSE),
      conditionalPanel(condition = 'input.H == true',
                       sliderInput("maxdepthXGB",
                                   label = "Maxdepth",
                                   min = 1, max = 30,
                                   value = 7,
                                   step = 1,
                                   p("Maximum depth of a tree")),
                       sliderInput("eta",
                                   label = "Learning Rate",
                                   min = 0, max = 1,
                                   value = 0.5,
                                   step = 0.1,
                                   p("Scale the contribution of each tree by a factor of (0, 1) when it is added to the current approximation.")),
                       sliderInput("nthreads",
                                   label = "Nthread",
                                   min = 1, max = 10,
                                   value = 7,
                                   step = 1,
                                   p("Limit the number of threads used by XGBoost at the predict operation")),
                       sliderInput("nrounds",
                                   label = "Nround",
                                   min = 10, max = 100,
                                   value = 33,
                                   step = 1,
                                   p("Limit the number of threads used by XGBoost at the predict operation")), 
                       sliderInput("subsample",
                                   label = "Subsample",
                                   min = 0, max = 1,
                                   value = 0.5,
                                   step = 0.1,
                                   p("Subsample ratio of the training instance.")),
                       sliderInput("min_child_weight",
                                   label = "Min Child Weight",
                                   min = 1, max = 100,
                                   value = 5,
                                   step = 1,
                                   p("Minimum sum of instance weight (hessian) needed in a child. "))
                     
      ),
      hr(),
      
      # Button

      actionButton("goButton", "Desicion Tree Root Mean Square Error!!"),
      hr(),
      actionButton("goButton1", "Random Forest Root Mean Square Error!!"),
      hr(),
      actionButton("goButton2", "XGBoosting Root Mean Square Error!!"),
      hr(),
      p("Click the buttons to calculate the Root Mean Square Errors !!!"),
  
     # Built with Shiny by RStudio
    br(),
    hr(),
    h4("Built with",
       img(src = "https://www.rstudio.com/wp-content/uploads/2014/04/shiny.png", height = "30px"),
       "by",
       img(src = "https://www.rstudio.com/wp-content/uploads/2014/07/RStudio-Logo-Blue-Gray.png", height = "30px"),
       "."),
    hr(),
    h3("Tigran Karamyan - Data Science For Business",
       hr(),
       img(src = "https://scontent.fevn1-2.fna.fbcdn.net/v/t1.0-9/32336816_1023526094464192_3863265427111018496_o.jpg?_nc_cat=100&_nc_ht=scontent.fevn1-2.fna&oh=fb3c7659769521244d6682dfbfe22dae&oe=5D5868D5", height = "315px"))
    
  ),
    mainPanel(
      h4("Summary"),
      verbatimTextOutput("summary"),
      h4("Observations"),
      tableOutput("view"),
      h4("Plot"),
      plotOutput("scatterplot"),
      h4("Model"),
      verbatimTextOutput("Model"),
      h4("Root Mean Square Error"),
      h5("Desicion Tree"),
      verbatimTextOutput("rmsedt"),
      h3("________________________"),
      h5("Random Forest"),
      verbatimTextOutput("rmserf"),
      h3("________________________"),
      h5("XGBoosting"),
      verbatimTextOutput("rmsexgb"),
      h3("________________________")
   

    )
  )
)

      

# Define server logic to display and download selected file ----
server <- function(input, output) {
  
  # Reactive value for selected dataset ----
  datasetInput <- reactive({
    switch(input$dataset,
           "Data" = Data,
           "Clear Data" = ClData)
  })
  
  
  # Generate a summary of the dataset ----
  output$summary <- renderPrint({
    dataset <- datasetInput()
    summary(dataset)
  })
  # Table of selected dataset ----
  output$view <- renderTable({
    dataset <- datasetInput()
    head(dataset)
  })
  
  # Create scatterplot

  scatterplot <- reactive({
    
    ## Plot 
    if(input$plot){
      return(scatterplot)
      }
    })
  
  output$scatterplot <- renderPlot({
    ggplot(data = ClData, aes_string(x = input$x, y = input$y)) +
      geom_point()
  })
  
  
  output$Model <- renderPrint({
    if (input$Model == "Desicion Tree") {
      BEST_DT <- rpart(as.formula(paste(input$Tar," ~ ",paste(input$Var,collapse="+"))),data = Data_train,
                       minsplit = input$minsplit, maxdepth = input$maxdepth, cp = input$cp)
      PRED_DT <- predict(BEST_DT, newdata = Data_test)
      print(BEST_DT$control)
      print(BEST_DT$call)
      print(BEST_DT$variable.importance)
      print(BEST_DT$frame)
      
    } else if (input$Model == "Random Forest") {
      BEST_RF <- randomForest(as.formula(paste(input$Tar," ~ ",paste(input$Var,collapse="+"))),data = Data_train,
                              mtry = input$mtry, nodesize = input$nodesize, ntree = input$ntree)
      PRED_RF <- predict(BEST_RF, newdata = Data_test)
      BEST_RF
    } else if (input$Model == "XGBoosting") {
      Target <- Data_train %>% pull(input$Tar)
      BEST_XGB <- xgboost(data = as.matrix(Data_train),label = Target,max.depth = input$maxdepthXGB,
                          eta = input$eta, nthreads = input$nthreads, nrounds = input$nrounds, subsample = input$subsample,
                          min_child_weight = input$min_child_weight, objective = "reg:linear")
      PRED_XGB <- predict(BEST_XGB, newdata = as.matrix(Data_test))
      BEST_XGB
    #  })
    }
  })
  
  # Calculate RMSE
  RMSE_DESICIONTREE <- eventReactive(input$goButton, {
    TarGet <- Data_test %>% pull(input$Tar)
    BEST_DT <- rpart(as.formula(paste(input$Tar," ~ ",paste(input$Var,collapse="+"))),data = Data_train,
                     minsplit = input$minsplit, maxdepth = input$maxdepth, cp = input$cp)
    PRED_DT <- predict(BEST_DT, newdata = Data_test)
    rmse(actual = TarGet, predicted = PRED_DT)
  })
  # Model
  output$rmsedt <- renderPrint({
    RMSE_DESICIONTREE()
    })
  
  RMSE_RANDOMFOREST <- eventReactive(input$goButton1, {
    TarGet1 <- Data_test %>% pull(input$Tar)
    BEST_RF <- randomForest(as.formula(paste(input$Tar," ~ ",paste(input$Var,collapse="+"))),data = Data_train,
                            mtry = input$mtry, nodesize = input$nodesize, ntree = input$ntree)
    PRED_RF <- predict(BEST_RF, newdata = Data_test)
    rmse(actual = TarGet1, predicted = PRED_RF)
  })
  # Model
  output$rmserf <- renderPrint({
    RMSE_RANDOMFOREST()
  })
  
  RMSE_XGBOOSTING <- eventReactive(input$goButton2, {
    Target <- Data_train %>% pull(input$Tar)
    TarGet2 <- Data_test %>% pull(input$Tar)
    BEST_XGB <- xgboost(data = as.matrix(Data_train),label = Target, max.depth = input$maxdepthXGB,
                        eta = input$eta, nthreads = input$nthreads, nrounds = input$nrounds, subsample = input$subsample,
                        min_child_weight = input$min_child_weight, objective = "reg:linear")
    PRED_XGB <- predict(BEST_XGB, newdata = as.matrix(Data_test))
    rmse(actual = TarGet2, predicted = PRED_XGB)
  })
  # Model
  output$rmsexgb <- renderPrint({
    RMSE_XGBOOSTING()
  })
  
}

# Create Shiny app ----
shinyApp(ui, server)
