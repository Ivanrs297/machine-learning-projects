import pandas as pd
import streamlit as st
import plotly.express as px

# Streamlit will perform internal magic so that the data will be downloaded only once and cached for future use.
@st.cache_data
def get_data():
    url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/visualisations/listings.csv"
    return pd.read_csv(url, low_memory=False)

df = get_data()

st.title("Streamlit Class")
st.markdown("Welcome")
st.header("Header")
st.markdown("*bold*")

st.dataframe(df.head())

st.code("""
@st.cache_data
def get_data():
    url = "http://data.insideairbnb.com/[...]"
    return pd.read_csv(url)
""", language="python")

#step 1 sort
st.subheader("In a table")
st.markdown("Following are the top five most expensive properties.")
st.write(df.query("price>=800").sort_values("price", ascending=False).head())


# step 2 query on df
st.header("Where are the most expensive properties located?")
st.subheader("On a map")
st.markdown("The following map shows the top 1% most expensive Airbnbs priced at $800 and above.")
st.map(df.query("price>=800")[["latitude", "longitude"]].dropna(how="any"))

# step 3 subset columns
st.subheader("Selecting a subset of columns")
default_cols = ["name", "host_name", "neighbourhood", "room_type", "price"]
cols = st.multiselect("Columns", df.columns.tolist(), default=default_cols)
st.dataframe(df[cols].head(10))

# step 4
# st.table displays a static table. 
# However, you cannot sort it by clicking a column header.
st.header("Average price by room type")
st.table(df.groupby("room_type").price.mean().reset_index().round(2).sort_values("price", ascending=False))


# step 5 Distributions - Sidebars
st.write("""Select a custom price range from the side bar to update the histogram """)
values = st.sidebar.slider("Price range", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (10., 500.))
f = px.histogram(df[df["price"].between(values[0], values[1])], x="price", nbins = 10)
f.update_xaxes(title="Price")
f.update_yaxes(title="No. of listings")
st.plotly_chart(f)
print("Values: ", values)

#step 6
# availability_365 indicates the number of days a property is
# available throughout the year. We examine summary
# statistics of availability_365 by neighborhood group.

#Radio button
neighborhood = st.radio("Neighborhood", df.neighbourhood_group.unique())
show_exp = st.checkbox("Include expensive listings")
show_exp = " and price<200" if not show_exp else ""
 
@st.cache_data
def get_availability(show_exp, neighborhood):
    return df.query(f"""neighbourhood_group==@neighborhood{show_exp}\
        and availability_365>0""").availability_365.describe(\
            percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

st.table(get_availability(show_exp, neighborhood))
