# Load necessary libraries
library(shiny)
library(tidyverse)
library(caret)
library(randomForest)
library(bslib)

# Define UI
ui <- fluidPage(
  theme = bs_theme(version = 4, bootswatch = "flatly"),  # Set a default theme
  tags$head(
    tags$style(HTML("
      body, html { height: 100%; }
      .container-fluid { height: 100vh; display: flex; flex-direction: column; }
      .row { flex-grow: 1; }
      .scrollable { overflow-y: auto; max-height: calc(100vh - 100px); }
      .btn-predict { background-color: #28a745; color: white; border-color: #28a745; }
      .btn-predict:hover { background-color: #218838; border-color: #1e7e34; }
      .result-box { background-color: #E6F2FF; padding: 10px; border-radius: 5px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); margin-top: 10px; }
      .input-group { margin-bottom: 10px; }
      h3 { margin-bottom: 15px; }
      .col-form-label { font-size: 0.9rem; }
      .form-control { font-size: 0.9rem; }
      .error-message { color: red; font-weight: bold; }
    "))
  ),
  titlePanel("❤️ NHS Early Risk Detection: Cardiovascular Module"),
  fluidRow(
    column(3,
           wellPanel(
             h4("🎨 App Settings"),
             selectInput("theme", "Choose Theme:",
                         choices = c("Default" = "flatly", "Cerulean" = "cerulean", "Cosmo" = "cosmo",
                                     "Cyborg" = "cyborg", "Darkly" = "darkly", "Journal" = "journal",
                                     "Lumen" = "lumen", "Paper" = "paper", "Readable" = "readable",
                                     "Sandstone" = "sandstone", "Simplex" = "simplex", "Slate" = "slate",
                                     "Spacelab" = "spacelab", "Superhero" = "superhero", "United" = "united",
                                     "Yeti" = "yeti"),
                         selected = "flatly")
           ),
           wellPanel(
             h4("📊 Prediction Results"),
             uiOutput("result")
           )
    ),
    column(9,
           div(class = "scrollable",
               h4("📝 Fill in the Required Information"),
               fluidRow(
                 column(4,
                        numericInput("age", "🎂 Age:", value = 50, min = 0, max = 120),
                        numericInput("cholesterol", "🩸 Cholesterol:", value = 200, min = 100, max = 500),
                        numericInput("bp_systolic", "🩺 BP Systolic:", value = 120, min = 50, max = 300),
                        numericInput("bp_diastolic", "🩺 BP Diastolic:", value = 80, min = 30, max = 200),
                        numericInput("heart_rate", "💓 Heart Rate:", value = 70, min = 40, max = 200),
                        selectInput("diabetes", "🍬 Diabetes:", choices = c("No" = "0", "Yes" = "1"))
                 ),
                 column(4,
                        selectInput("family_history", "👪 Family History:", choices = c("No" = "0", "Yes" = "1")),
                        selectInput("smoking", "🚬 Smoking:", choices = c("No" = "0", "Yes" = "1")),
                        selectInput("obesity", "🍔 Obesity:", choices = c("No" = "0", "Yes" = "1")),
                        selectInput("alcohol_consumption", "🍺 Alcohol Consumption:", choices = c("No" = "0", "Yes" = "1")),
                        numericInput("exercise_hours", "🏃‍♂️ Exercise Hours Per Week:", value = 3, min = 0, max = 20),
                        selectInput("diet", "🥗 Diet:", choices = c("Healthy", "Average", "Unhealthy"))
                 ),
                 column(4,
                        selectInput("previous_heart_problems", "💔 Previous Heart Problems:", choices = c("No" = "0", "Yes" = "1")),
                        selectInput("medication_use", "💊 Medication Use:", choices = c("No" = "0", "Yes" = "1")),
                        numericInput("bmi", "⚖️ BMI:", value = 25, min = 10, max = 50),
                        numericInput("triglycerides", "🩸 Triglycerides:", value = 150, min = 50, max = 500),
                        numericInput("sleep_hours", "😴 Sleep Hours Per Day:", value = 7, min = 0, max = 12),
                        selectInput("sex", "⚧️ Sex:", choices = c("Male", "Female"))
                 )
               ),
               actionButton("predict", "🔮 Predict", class = "btn-predict"),
               textOutput("error_message")
           )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Apply the selected theme
  observe({
    theme <- bs_theme(version = 4, bootswatch = input$theme)
    session$setCurrentTheme(theme)
  })
  
  # Load and preprocess data
  data <- reactive({
    # Load dataset
    data <- read.csv("Data/heart_attack_prediction_dataset.csv")
    
    # Convert categorical variables to factors
    data$Sex <- as.factor(data$Sex)
    data$Diabetes <- as.factor(data$Diabetes)
    data$Family.History <- as.factor(data$Family.History)
    data$Smoking <- as.factor(data$Smoking)
    data$Obesity <- as.factor(data$Obesity)
    data$Alcohol.Consumption <- as.factor(data$Alcohol.Consumption)
    data$Previous.Heart.Problems <- as.factor(data$Previous.Heart.Problems)
    data$Medication.Use <- as.factor(data$Medication.Use)
    data$Diet <- as.factor(data$Diet)
    data$Heart.Attack.Risk <- as.factor(data$Heart.Attack.Risk)
    
    # Feature engineering
    data$BP_systolic <- as.numeric(sub("/.*", "", data$Blood.Pressure))
    data$BP_diastolic <- as.numeric(sub(".*/", "", data$Blood.Pressure))
    data <- select(data, -Blood.Pressure)  # Drop old Blood Pressure column
    
    return(data)
  })
  
  # Train model
  rf_model <- reactive({
    data_train <- data()
    
    # Select only the relevant columns for training
    train_data <- data_train %>%
      select(Age, Cholesterol, BP_systolic, BP_diastolic, Heart.Rate,
             Diabetes, Family.History, Smoking, Obesity,
             Alcohol.Consumption, Exercise.Hours.Per.Week, Previous.Heart.Problems, Medication.Use,
             BMI, Triglycerides, Sleep.Hours.Per.Day,
             Sex, Diet, Heart.Attack.Risk)
    
    # Split data into training and test sets
    set.seed(123)
    trainIndex <- createDataPartition(train_data$Heart.Attack.Risk, p = 0.8, list = FALSE)
    train_data <- train_data[trainIndex, ]
    test_data <- train_data[-trainIndex, ]
    
    # Scale numeric features
    preprocessParams <- preProcess(train_data, method = c("center", "scale"))
    train_data <- predict(preprocessParams, train_data)
    test_data <- predict(preprocessParams, test_data)
    
    # Train Random Forest model
    rf_model <- randomForest(Heart.Attack.Risk ~ ., data = train_data)
    
    return(list(model = rf_model, preprocess = preprocessParams, factor_levels = list(
      Sex = levels(train_data$Sex),
      Diet = levels(train_data$Diet),
      Diabetes = levels(train_data$Diabetes),
      Family.History = levels(train_data$Family.History),
      Smoking = levels(train_data$Smoking),
      Obesity = levels(train_data$Obesity),
      Alcohol.Consumption = levels(train_data$Alcohol.Consumption),
      Previous.Heart.Problems = levels(train_data$Previous.Heart.Problems),
      Medication.Use = levels(train_data$Medication.Use)
    )))
  })
  
  observeEvent(input$predict, {
    if (input$age <= 0) {
      output$error_message <- renderText({
        "Error: Age must be greater than 0."
      })
      output$result <- renderUI({
        div(class = "result-box", h4("Please correct the age input before predicting."))
      })
    } else {
      output$error_message <- renderText({ "" })
      
      model_info <- rf_model()
      rf_model <- model_info$model
      preprocessParams <- model_info$preprocess
      factor_levels <- model_info$factor_levels
      
      # Create new data from user inputs
      new_data <- data.frame(
        Age = input$age,
        Cholesterol = input$cholesterol,
        BP_systolic = input$bp_systolic,
        BP_diastolic = input$bp_diastolic,
        Heart.Rate = input$heart_rate,
        Diabetes = factor(input$diabetes, levels = factor_levels$Diabetes),
        Family.History = factor(input$family_history, levels = factor_levels$Family.History),
        Smoking = factor(input$smoking, levels = factor_levels$Smoking),
        Obesity = factor(input$obesity, levels = factor_levels$Obesity),
        Alcohol.Consumption = factor(input$alcohol_consumption, levels = factor_levels$Alcohol.Consumption),
        Exercise.Hours.Per.Week = input$exercise_hours,
        Diet = factor(input$diet, levels = factor_levels$Diet),
        Previous.Heart.Problems = factor(input$previous_heart_problems, levels = factor_levels$Previous.Heart.Problems),
        Medication.Use = factor(input$medication_use, levels = factor_levels$Medication.Use),
        BMI = input$bmi,
        Triglycerides = input$triglycerides,
        Sleep.Hours.Per.Day = input$sleep_hours,
        Sex = factor(input$sex, levels = factor_levels$Sex)
      )
      
      # Pre-process new data
      new_data_processed <- predict(preprocessParams, new_data)
      
      # Predict using Random Forest model
      prediction <- predict(rf_model, new_data_processed)
      
      # Display the prediction
      output$result <- renderUI({
        risk <- ifelse(prediction == "1", "High Risk ⚠️", "Low Risk ✅")
        result_text <- paste("Predicted Heart Attack Risk:", risk)
        
        div(
          class = "result-box",
          h4(result_text)
        )
      })
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)