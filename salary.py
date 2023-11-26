import streamlit as st
from streamlit_option_menu import option_menu
import requests
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
from plotly import graph_objs as go

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.filterwarnings("ignore")

#-----------Web page setting-------------------#
page_title = "Developer Salary Predition App"
page_icon = ":robot"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)


# Set up Menu/Navigation Bar
selected = option_menu(
    menu_title = "SalaryPredictionApp",
    options = ['Home', 'Explore', 'Prediction', 'Contact'],
    icons = ["house-fill", "book-half", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

# Load and clean data
@st.cache_data
def load_data():
    # Define the URL of the zip file
    url = "https://cdn.stackoverflow.co/files/jo7n4k8s/production/49915bfd46d0902c3564fd9a06b509d08a20488c.zip/stack-overflow-developer-survey-2023.zip"
    response = requests.get(url) # Get data
    zip_data = BytesIO(response.content) # Get Zip file contents
    
    # Use the zipfile library to extract the contents of the zip file
    with zipfile.ZipFile(zip_data, "r") as zip_file:
        # List the contents of the zip file (optional)
        print("Contents of the zip file:")
        print(zip_file.namelist())

        # Extract the specific CSV file you want (e.g., survey_results_public.csv)
        extracted_file = zip_file.extract("survey_results_public.csv")
        file_structure = zip_file.extract("survey_results_schema.csv")
    
    # Read extracted file
    df = pd.read_csv(extracted_file)

    # Data Cleaning
    # Select columns to concentrate on
    selected_col = ['Age','Country', 'EdLevel', 
                    'YearsCodePro', 'Employment', 'DevType',
                    'AISelect', 'AISent', 'AIBen', 'WorkExp','Industry',
                    'RemoteWork', 'ConvertedCompYearly']
    data = df[selected_col]
    data = data.rename({'ConvertedCompYearly': 'Salary'}, axis = 1)
    # Take null values out of the `Salary` column
    data = data[data['Salary'].notnull()]
    #Drop NaN values
    data = data.dropna()

    # Clean `Age` column
    data['Age'] = data['Age'].str.replace(' years old', '', regex=False)
    data['Age'].replace('65 years or older', 'Above 65', inplace = True)

    #Clean the `Country` column
    africa = ['Senegal', 'Cape Verde', 'Madagascar', 'Niger', 'Lesotho', 'Swaziland',
          'United Republic of Tanzania', 'Algeria', 'Mauritania', 'Rwanda', 'Namibia', 'Angola',
          'Togo', 'Egypt', 'Zimbabwe', 'Nomadic', 'Kenya', 'Tunisia', 'Ethiopia', 'Nigeria',
          'Benin','Somalia', 'Morocco', "Côte d'Ivoire", 'Zambia', 'Ghana', 'Uganda',
          'Mozambique', 'Gabon', 'Malawi','Cameroon', 'Mauritius', 'Namibia']


    south_america = ['Bolivia', 'Argentina', 'Peru', 'Chile', 'Guyana', 'Ecuador', 'Paraguay', 'Suriname',
            'Uruguay', 'Venezuela, Bolivarian Republic of...']

    oceania = ['Fiji', 'Palau']

    north_america = ['Barbados', 'Belize', 'Costa Rica', 'Cuba', 'Dominican Republic', 'El Salvador', 'Guatemala',
            'Honduras', 'Jamaica', 'Nicaragua', 'Panama', 'Saint Lucia', 'Trinidad and Tobago']

    europe = ['Albania', 'Andorra', 'Belarus', 'Slovenia', 'Slovakia', 'Serbia', 'Russian Federation',
            'The former Yugoslav Republic of Macedonia', 'Montenegro', 'Republic of Moldova',
            'Malta', 'Luxembourg', 'Lithuania', 'Liechtenstein', 'Latvia', 'Kosovo', 'Ireland',
            'Iceland', 'Hungary', 'Greece', 'Estonia', 'Croatia', 'Bulgaria', 'Bosnia and Herzegovina']

    asia = ['Yemen', 'Viet Nam', 'United Arab Emirates', 'Turkmenistan', 'Thailand', 'Taiwan',
            'Syrian Arab Republic', 'Sri Lanka', 'Republic of Korea', 'South Korea', 'Singapore',
            'Saudi Arabia', 'Qatar', 'Philippines', 'Palestine', 'Oman', 'Nepal', 'Myanmar',
            'Mongolia', 'Maldives', 'Malaysia', 'Lebanon', 'Kyrgyzstan', 'Kuwait', 'Kazakhstan',
            'Jordan', 'Japan', 'Iraq', 'Iran, Islamic Republic of...', 'Indonesia', 'Georgia',
            'Cambodia', 'China', 'Cyprus', 'Brunei Darussalam', 'Bangladesh', 'Bahrain','Azerbaijan',
            'Armenia', 'Afghanistan', 'Hong Kong (S.A.R.)', 'Uzbekistan']
    
    data['Country'].replace(africa, 'Other African Country', inplace=True)
    data['Country'].replace(south_america, 'Other South American Country', inplace=True)
    data['Country'].replace(north_america, 'Other North American Country', inplace=True)
    data['Country'].replace(oceania, 'Other Oceanian Country', inplace=True)
    data['Country'].replace(europe, 'Other European Country', inplace=True)
    data['Country'].replace(asia, 'Other Asian Country', inplace=True)
    data['Country'].replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom', inplace=True)
    data['Country'].replace('United States of America', 'USA', inplace=True)

    # Clean `EdLevel` column
    def clean_edu(x):
        if 'Master’s degree' in x:
            return 'Master’s degree'
        if 'Bachelor’s degree' in x:
            return 'Bachelor’s degree'
        if 'Professional degree' in x or 'doctoral degree' in x:
            return 'Post Grad degree'
        return 'Other certifications'
    data['EdLevel'] = data['EdLevel'].apply(clean_edu)

    # Clean `YearsCodePro` column
    data['YearsCodePro'].replace('Less than 1 year', 0.5, inplace=True)
    data['YearsCodePro'].replace('More than 50 years', 50, inplace=True)
    data['YearsCodePro'] = pd.to_numeric(data['YearsCodePro'], errors='coerce')
    data['YearsCodePro'] = data['YearsCodePro'].astype(float)

    # Clean `Employment` column
    full_time = ["Employed, full-time",
        "Employed, full-time;Independent contractor, freelancer, or self-employed",
        "Employed, full-time;Independent contractor, freelancer, or self-employed;Employed, part-time",
        "Employed, full-time;Employed, part-time",
        "Employed, full-time;Independent contractor, freelancer, or self-employed;Retired",
        "Employed, full-time;Retired"]

    part_time = ['Employed, part-time',
                'Employed, part-time;Retired']

    freelancer = ['Independent contractor, freelancer, or self-employed',
                'Independent contractor, freelancer, or self-employed;Retired',
                'Independent contractor, freelancer, or self-employed;Employed, part-time']
    data['Employment'].replace(full_time, 'Employed full time', inplace=True)
    data['Employment'].replace(part_time, 'Employed part time', inplace=True)
    data['Employment'].replace(freelancer, 'Freelancer/Independent contractor', inplace=True)

    # Clean `DevType` column
    data['DevType'].replace('Senior Executive (C-Suite, VP, etc.)', 'Senior Executive', inplace=True)
    data['DevType'].replace('Developer, full-stack', 'Full Stack Developer', inplace=True)
    data['DevType'].replace('Developer, back-end', 'Back-end Developer', inplace=True)
    data['DevType'].replace('Developer, front-end', 'Front-end Developer', inplace=True)
    data['DevType'].replace('System administrator', 'System Administrator', inplace=True)
    data['DevType'].replace('Database administrator', 'Database Administrator', inplace=True)
    data['DevType'].replace('Developer, mobile', 'Mobile Developer', inplace=True)
    data['DevType'].replace('Developer, game or graphics', 'Game Developer', inplace=True)
    data['DevType'].replace(['Developer, desktop or enterprise applications', 'Developer, embedded applications or devices'],
                            'Application Developer', inplace=True)
    data['DevType'].replace('Academic researcher', 'Academic Researcher', inplace=True)
    data['DevType'].replace('Other (please specify):', 'Other', inplace=True)
    data['DevType'].replace('Engineering manager', 'Engineering Manager', inplace=True)
    data['DevType'].replace('Engineer, data', 'Data Engineer', inplace=True)
    data['DevType'].replace('Data scientist or machine learning specialist', 'Data Science/ML Specialist', inplace=True)
    data['DevType'].replace('Engineer, site reliability', 'Site Reliability Engineer', inplace=True)
    data['DevType'].replace('Cloud infrastructure engineer', 'Cloud Engineer', inplace=True)
    data['DevType'].replace('DevOps specialist', 'DevOps Specialist', inplace=True)
    data['DevType'].replace('Research & Development role', 'Research & Development Role', inplace=True)
    data['DevType'].replace('Blockchain', 'Blockchain Developer', inplace=True)
    data['DevType'].replace('Product manager', 'Product Manager', inplace=True)
    data['DevType'].replace('Project manager', 'Project Manager', inplace=True)
    data['DevType'].replace('Security professional', 'Security Professional', inplace=True)
    data['DevType'].replace('Data or business analyst', 'Data/Business Analyst', inplace=True)
    data['DevType'].replace('Marketing or sales professional', 'Marketing/Sales Professional', inplace=True)
    data['DevType'].replace('Developer, QA or test', 'Testing Developer', inplace=True)

    # Clean `AISelect` column
    data['AISelect'].replace('No, but I plan to soon', 'No', inplace=True)

    # Create 'WorkExpGroup' column
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, float('inf')]  # Define the bin edges
    labels = ['Below 1 year', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']
    data['WorkExpGroup'] = pd.cut(data['WorkExp'], bins=bins, labels=labels, include_lowest=True)

    # Clean `Industry` column
    data['Industry'].replace('Information Services, IT, Software Development, or other Technology', 'I.T.', inplace=True)
    data['Industry'].replace('Higher Education', 'Education', inplace=True)
    data['Industry'].replace('Manufacturing, Transportation, or Supply Chain', 'Manufacturing/Transportation', inplace=True)

    # Clean `RemoteWork` column
    data['RemoteWork'].replace('Hybrid (some remote, some in-person)', 'Hybrid', inplace=True)

    # Clean `Salary` column
    data = data[data['Salary'] <= 500000]
    data = data[data['Salary'] >= 10000]

    # Return `data`
    return data

data = load_data() # Load the data

# Set `Home` page
if selected == "Home":
    st.image("stack-overflow.png", caption="2023 Stack Overflow Survey", use_column_width=True)
    st.write("""MovieRecommender is a web app that analyzes the 2023 Stack Overflow Developer Survey data and uses Supervised Machine Learning algorithms to predict salary using some user input details.""")
    st.markdown("""The data is taken from [Stack Overflow Annual Developer Survey](https://insights.stackoverflow.com/survey).""")    

# Set `Explore` page
if selected == "Explore":
    st.write("""## Explore the Demography of Loan Defaulter Data""")
    tab1, tab2, tab3 = st.tabs(["Developers Demography Analysis", "Salary Analysis", "AI Usage Analysis"])

    with tab1:
        chart_opt = ["Age Group Representation", 
                     "Country Representation", 
                     "Developers' Level of Education", 
                     "Type of Employment",
                     "Type of Developer",
                     "Industry Working In"
                     "Working Type",
                     "Work Experience Group Prepsentation"]
        
        chart = st.selectbox("Select analysis:", chart_opt)

        if chart == "Age Group Representation":
            mask = data['Age'].value_counts().sort_values()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Age Group Representation", xaxis_title="Number of Developers", yaxis_title="Age Group")

            st.plotly_chart(fig)
        
        if chart == "Country Representation":
            mask = data['Country'].value_counts().sort_values().tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Top 10 Most Represented Countries", xaxis_title="Number of Developers", yaxis_title="Country")

            st.plotly_chart(fig)

        if chart == "Developers' Level of Education":
            mask = data['EdLevel'].value_counts().sort_values()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Level Of Education Representation", xaxis_title="Number of Developers", yaxis_title="Level Of Education")

            st.plotly_chart(fig)
        
        if chart == "Type of Employment":
            mask = data['Employment'].value_counts().sort_values()
            
            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Type of Employment Representation", xaxis_title="Number of Developers", yaxis_title="Employment Type")
            
            st.plotly_chart(fig)
        
        if chart == "Type of Developer":
            mask = data['DevType'].value_counts().sort_values().tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Top 10 Most Represented Developer Type", xaxis_title="Number of Developers", yaxis_title="Type of Developer")

            st.plotly_chart(fig)

        if chart == "Industry Working In":
            mask = data['Industry'].value_counts().sort_values() #.tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Industry Representation of Developers", xaxis_title="Number of Developers", yaxis_title="Industry")
            
            st.plotly_chart(fig)

        if chart == "Working Type":
            mask = data['RemoteWork'].value_counts().sort_values() #.tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Working Condition Representation of Developers", xaxis_title="Number of Developers", yaxis_title="Working Condition")

            st.plotly_chart(fig)

        if chart == "Work Experience Group Prepsentation":
            mask = data['WorkExpGroup'].value_counts().sort_values()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Work Experience Group Representation", xaxis_title="Number of Developers", yaxis_title="Work Experience Group")

            st.plotly_chart(fig)

    with tab2:
        chart_opt = ["Years of Coding Experience VS Salary",
                     "Years of Working Experience VS Salary",
                     "Age Group VS Salary", 
                     "Country VS Salary", 
                     "Developers' Level of Education VS Salary", 
                     "Type of Employment VS Salary",
                     "Type of Developer VS Salary",
                     "Industry Working In VS Salary"
                     "Working Type VS Salary",
                     "Work Experience Group VS Salary"]
        
        chart = st.selectbox("Select analysis:", chart_opt)

        if chart == "Years of Coding Experience VS Salary":
            fig = go.Figure([go.Scatter(x=data['YearsCodePro'], y=data['Salary'], mode='markers', marker=dict(color='blue'))])
            fig.update_layout(title_text="Relationship between Years of Coding Experience and Salary", xaxis_title="Years of Coding Experience", yaxis_title="Salary (US$)")

            st.plotly_chart(fig)

        if chart == "Years of Working Experience VS Salary":
            fig = go.Figure([go.Scatter(x=data['WorkExp'], y=data['Salary'], mode='markers', marker=dict(color='blue'))])
            fig.update_layout(title_text="Relationship between Years of Work Experience and Salary", xaxis_title="Years of Work Experience", yaxis_title="Salary (US$)")

            st.plotly_chart(fig)

        if chart == "Age Group VS Salary":
            mask = data.groupby(['Age'])['Salary'].mean()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution Based on Age Group", xaxis_title="Mean Salary (US$)", yaxis_title="Age Group")

            st.plotly_chart(fig)
        
        if chart == "Country VS Salary":
            mask = data.groupby(['Country'])['Salary'].mean().tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Top 10 Highest Paying Countries", xaxis_title="Mean Salary (US$)", yaxis_title="Country")

            st.plotly_chart(fig)
        
        if chart == "Developers' Level of Education VS Salary":
            mask = data.groupby(['EdLevel'])['Salary'].mean()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution Based on Qualification", xaxis_title="Mean Salary (US$)", yaxis_title="Qualification")

            st.plotly_chart(fig)
        
        if chart == "Type of Employment VS Salary":
            mask = data.groupby(['Employment'])['Salary'].mean()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution Based Type of Employment", xaxis_title="Mean Salary (US$)", yaxis_title="Type of Employment")

            st.plotly_chart(fig)

        if chart == "Type of Developer VS Salary":
            mask = data.groupby(['DevType'])['Salary'].mean().sort_values().tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution of Top 10 Type of Developers", xaxis_title="Mean Salary (US$)", yaxis_title="Type of Developer")

            st.plotly_chart(fig)

        if chart == "Industry Working In VS Salary":
            mask = data.groupby(['Industry'])['Salary'].mean().sort_values().tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Top 10 Highest Paying Industries", xaxis_title="Mean Salary (US$)", yaxis_title="Industry")

            st.plotly_chart(fig)

        if chart == "Working Type VS Salary":
            mask = data.groupby(['RemoteWork'])['Salary'].mean()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution Based on Working Type",xaxis_title="Mean Salary (US$)", yaxis_title="Work Type")
            
            st.plotly_chart(fig)

        if chart == "Work Experience Group VS Salary":
            mask = data.groupby(['WorkExpGroup'])['Salary'].mean()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Salary Distribution Based on Work Experience", xaxis_title="Mean Salary (US$)", yaxis_title="Work Experience Group")

            st.plotly_chart(fig)

    with tab2:
        chart_opt = ["AI Users",
                     "AI Usage Semtiment",
                     "AI Usage Benefit"]
        
        chart = st.selectbox("Select analysis:", chart_opt)

        if chart == "AI Users":
            mask = data['AISelect'].value_counts().sort_values()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Developers'Usage of AI", xaxis_title="Number of Developers", yaxis_title="AI Usage")

            st.plotly_chart(fig)
        
        if chart == "AI Usage Semtiment":
            mask = data['AISent'].value_counts().sort_values()

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="AI Usage Sentiment of Developers", xaxis_title="Number of Developers", yaxis_title="AI Usage Sentiment")

            st.plotly_chart(fig)

        if chart == "AI Usage Benefit":
            mask = data['AIBen'].value_counts().sort_values() #.tail(10)

            fig = go.Figure([go.Bar(y=mask.index, x=mask.values, orientation='h')])
            fig.update_layout(title_text="Developers' View on AI Usage Benefit", xaxis_title="Number of Developers", yaxis_title="AI Usage Benefit")

            st.plotly_chart(fig)

# Set `Prediction` page
if selected == "Prediction":
    # Split data
    X = data.drop(['AISent', 'AIBen', 'WorkExp', 'Salary'], axis = 1)
    y = data['Salary']

    # Set encoders
    age_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    edu_encoder = LabelEncoder()
    emp_encoder = LabelEncoder()
    dev_encoder = LabelEncoder()
    ai_encoder = LabelEncoder()
    industry_encoder = LabelEncoder()
    rem_encoder = LabelEncoder()
    exp_encoder = LabelEncoder()

    # Encode variables
    X['Age'] = age_encoder.fit_transform(X['Age'])
    X['Country'] = country_encoder.fit_transform(X['Country'])
    X['EdLevel'] = edu_encoder.fit_transform(X['EdLevel'])
    X['Employment'] = emp_encoder.fit_transform(X['Employment'])
    X['AISelect'] = ai_encoder.fit_transform(X['AISelect'])
    X['Industry'] = industry_encoder.fit_transform(X['Industry'])
    X['RemoteWork'] = rem_encoder.fit_transform(X['RemoteWork'])
    X['WorkExpGroup'] = exp_encoder.fit_transform(X['WorkExpGroup'])

    # Create model
    dec = DecisionTreeRegressor()
    dec.fit(X, y)
    # accuracy = dec.score(X, y)

    # Building the app to show `Prediction`
    st.title("Bank Loan Default Prediction App")
    st.write("""#### Imput the following details to predict loan defaulting.""")

    # Define some terms
    age_group = ['25-34', '35-44', '55-64', '18-24', '45-54', 'Under 18', 'Above 65', 'Prefer not to say']
    countries = ['USA', 'Other Asian Country', 'United Kingdom', 'Netherlands',
       'Germany', 'France', 'Spain', 'South Africa', 'Italy',
       'Other European Country', 'Other African Country', 'Brazil',
       'Norway', 'Turkey', 'Sweden', 'India', 'Poland', 'Austria',
       'Romania', 'Canada', 'Belgium', 'Israel', 'Ukraine', 'Finland',
       'Other North American Country', 'Switzerland', 'Denmark',
       'Portugal', 'Australia', 'Czech Republic',
       'Other South American Country', 'Pakistan', 'Colombia', 'Mexico',
       'New Zealand', 'Other Oceanian Country', 'Isle of Man']
    edu_level = ['Bachelor’s degree', 'Other certifications', 'Master’s degree', 'Post Grad degree']
    employment = ['Employed full time', 'Freelancer/Independent contractor', 'Employed part time']
    dev_type = ['Senior Executive', 'Full Stack Developer', 'Back-end Developer',
       'Testing Developer', 'Front-end Developer', 'System Administrator',
       'Mobile Developer', 'Application Developer', 'Academic Researcher',
       'Other', 'Engineering Manager', 'Data Engineer',
       'Data Science/ML Specialist', 'Database Administrator',
       'Site Reliability Engineer', 'Cloud Engineer', 'DevOps Specialist',
       'Research & Development Role', 'Blockchain Developer',
       'Developer Advocate', 'Product Manager', 'Project Manager',
       'Security Professional', 'Hardware Engineer',
       'Developer Experience', 'Data/Business Analyst',
       'Marketing/Sales Professional', 'Designer', 'Game Developer',
       'Scientist', 'Student', 'Educator']
    ai_usage = ['Yes', 'No']
    industry = ['I.T.', 'Other', 'Financial Services',
       'Manufacturing/Transportation', 'Retail and Consumer Services',
       'Education', 'Legal Services', 'Healthcare', 'Oil & Gas',
       'Wholesale', 'Advertising Services', 'Insurance']
    remote = ['Remote', 'Hybrid', 'In-person']
    work_exp = ['Below 1 year','1-5', '6-10','11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45']

    # Set user inputs
    age_group = st.selectbox("Age: ", age_group)
    countries = st.selectbox("Country of interest:", countries)
    edu_level = st.selectbox("Level of education:", edu_level)
    year_code  = st.text_input("Years of coding experience:", placeholder = "Enter only numbers(eg: 5)")
    employment = st.selectbox("Type of employment: ", employment)
    dev_type = st.selectbox("Type of developer:", dev_type)
    ai_usage = st.selectbox("Use of AI in coding:", ai_usage)
    industry = st.selectbox("Industry of interest: ", industry)
    remote = st.selectbox("Working type:", remote)
    work_exp = st.selectbox("Years of working experience:", work_exp)
    estimate_btn = st.button("Estimate Salary")

    # Get salary prediction
    if estimate_btn:
        X_test = np.array([[
            age_group,
            countries,
            edu_level,
            year_code,
            employment,
            dev_type,
            ai_usage,
            industry,
            remote,
            work_exp
        ]])

        # Get prediction
        X_test[:, 0] = age_encoder.transform(X_test[:, 0])
        X_test[:, 1] = country_encoder.transform(X_test[:, 1])
        X_test[:, 2] = edu_encoder.transform(X_test[:, 2])
        X_test[:, 4] = emp_encoder.transform(X_test[:, 4])
        X_test[:, 5] = dev_encoder.transform(X_test[:, 5])
        X_test[:, 6] = ai_encoder.transform(X_test[:, 6])
        X_test[:, 7] = industry_encoder.transform(X_test[:, 7])
        X_test[:, 8] = rem_encoder.transform(X_test[:, 8])
        X_test[:, 9] = exp_encoder.transform(X_test[:, 9])
        X_test = X_test.astype(float)

        salary = dec.predict(X_test)
        st.subheader(f"Estimated salary per year: US${salary[0]:,.2f}")

# Set `Contact` page
if selected == "Contact":
    # Contact Web App
    st.write("""### Get in touch""")
    st.markdown("""Email: [Link](mailto:gamahrichard5@gmail.com).""")

    st.markdown("""GitHub: [Link](https://github.com/SirGamah/).""")

    st.markdown("""WhatsApp: [Link](https://wa.me/233542124371).""")
