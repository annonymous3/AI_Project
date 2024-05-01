from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("supermarket_sales.csv")

# Convert dataset to HTML table format
dataset_html = dataset.to_html(index=False)


def generate_visualization(visualization_type):
    if visualization_type == 'product_distribution':
        return generate_product_distribution_plot()
    elif visualization_type == 'profitability':
        return generate_profitability_plot()
    elif visualization_type == 'revenue':
        return generate_revenue_plot()
    elif visualization_type == 'sales_volume':
        return generate_sales_volume_plot()
    elif visualization_type == 'sales_volume_by_gender':
        return sales_volume_segmented_by_gender_plot()
    elif visualization_type == 'monthly_income':
        return generate_monthly_income_plot()
    elif visualization_type == 'gross_income_by_gender':
        return generate_gross_income_by_gender_plot()
    elif visualization_type == 'monthly_gross_income':
        return generate_monthly_gross_income_plot()
    elif visualization_type == 'total_gross_income_by_branch':
        return generate_total_gross_income_by_branch_plot()
    elif visualization_type == 'average_ratings_by_product_lines':
        return average_ratings_by_product_lines()
    elif visualization_type == 'product_lines_gross_income':
        return product_lines_gross_income()
    elif visualization_type == 'average_ratings_vs_sales_volume':
        return average_ratings_vs_sales_volume()
    elif visualization_type == 'cogs_and_gross_income':
        return cogs_gross_income()
    elif visualization_type == 'correlation_heatmap':
        return correlation_heatmap()
    else:
        return None, None


def correlation_heatmap():
    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
    numeric_data = dataset[numeric_columns]
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
    <p>The provided dataset contains information on unit price, quantity sold, gross income, and rating for various 
    products. Let's explore some insights:</p>

    <ul>
      <li><strong>Correlation between Unit Price and Gross Income:</strong></li>
      <p>Higher unit prices may lead to higher gross income if the quantity sold remains constant. Analyzing this 
      relationship can help identify pricing strategies to maximize revenue.</p>
      
      <li><strong>Impact of Quantity Sold on Gross Income:</strong></li>
      <p>The quantity sold directly affects gross income. Understanding how changes in quantity impact income can inform
       inventory management and sales forecasting.</p>
      
      <li><strong>Relationship between Rating and Gross Income:</strong></li>
      <p>Products with higher ratings may attract more customers and result in higher gross income. Evaluating this 
      correlation can guide product development and marketing efforts to enhance customer satisfaction and revenue.</p>
    </ul>
    
    <p>By analyzing these relationships, businesses can make informed decisions to optimize pricing strategies, manage 
    inventory effectively, and enhance product quality to drive overall profitability.</p>

                    '''
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Correlation Heatmap">', explanation


def cogs_gross_income():
    sns.jointplot(x='cogs', y='gross income', data=dataset, kind='reg')
    plt.xlabel('Cost of Goods Sold (COGS)')
    plt.ylabel('Gross Income')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
    .
                '''
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Cost of Goods Sold and Gross income">', explanation


def average_ratings_vs_sales_volume():
    avg_rating = dataset.groupby('Product line')['Rating'].mean()
    sales_volume = dataset.groupby('Product line')['Quantity'].sum()

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_rating, sales_volume, c='skyblue')
    plt.title('Average Rating vs. Sales Volume')
    plt.xlabel('Average Rating')
    plt.ylabel('Sales Volume')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
            <p>The data presents the average ratings and quantity of products sold for various product lines. Let's 
            delve into the insights derived from this information:</p>

            <ul>
              <li><strong>Positive Correlation between Average Rating and Quantity Sold:</strong></li>
              <ul>
                <li>Products in the "Food and Beverages" category boast the highest average rating of 7.11, indicating 
                high customer satisfaction. This satisfaction likely translates into increased demand, as evidenced by 
                the substantial quantity sold.</li>
                <li>Similarly, "Health and Beauty" products also exhibit a commendable average rating of 7.00, 
                contributing to a respectable quantity sold. This suggests that perceived quality in this category 
                drives consumer interest and sales.</li>
                <li>Conversely, "Home and Lifestyle" products, despite having a slightly lower average rating of 6.84, 
                maintain a notable quantity sold. This hints at factors beyond average rating, such as utility or 
                trendiness, influencing consumer purchasing decisions.</li>
              </ul>
              <li><strong>Potential for Improvement in Electronic Accessories:</strong></li>
              <ul>
                <li>While "Electronic Accessories" garner a decent average rating of 6.92, the quantity sold doesn't 
                reflect the same level of consumer demand as observed in other categories. This indicates an opportunity
                 for improvement, perhaps through enhancing product features or addressing customer pain points.</li>
              </ul>
              <li><strong>Exploring Customer Preferences in Fashion Accessories and Sports and Travel:</strong></li>
              <ul>
                <li>"Fashion Accessories" and "Sports and Travel" products demonstrate comparable average ratings and 
                quantity sold. Further analysis could uncover nuanced preferences within these categories, guiding 
                marketing strategies or product diversification efforts to cater to specific consumer segments.</li>
              </ul>
              <li><strong>Strategic Considerations for Business Growth:</strong></li>
              <ul>
                <li>Understanding the relationship between average rating and quantity sold enables businesses to 
                strategize effectively. By focusing on enhancing product quality, customer experience, or marketing 
                initiatives, companies can capitalize on high-performing categories and address weaknesses to drive 
                overall growth.</li>
              </ul>
            </ul>
            
            <p>In conclusion, this analysis underscores the significance of both customer satisfaction and sales volume 
            in shaping business success. By leveraging insights from average ratings and quantity sold, companies can 
            refine their product offerings, tailor marketing strategies, and optimize operational efforts to meet 
            consumer needs and drive sustainable growth.</p>
                        '''
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Average Rating vs. Sales Volume">', explanation


def product_lines_gross_income():
    monthly_income = dataset.groupby('Product line')['gross income'].sum().sort_values()
    sns.lineplot(x=monthly_income.index, y=monthly_income.values)
    plt.title('Product Line Gross Income')
    plt.xlabel('Product line')
    plt.xticks(rotation=45)
    plt.ylabel('Gross Income')
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
        <p><strong>Gross Income by Product Line:</strong></p>

        <p>The table below displays the gross income for each product line:</p>
        
        <table>
          <tr>
            <th>Product Line</th>
            <th>Gross Income ($)</th>
          </tr>
          <tr>
            <td>Health and Beauty</td>
            <td>2342.5590</td>
          </tr>
          <tr>
            <td>Home and Lifestyle</td>
            <td>2564.8530</td>
          </tr>
          <tr>
            <td>Fashion Accessories</td>
            <td>2585.9950</td>
          </tr>
          <tr>
            <td>Electronic Accessories</td>
            <td>2587.5015</td>
          </tr>
          <tr>
            <td>Sports and Travel</td>
            <td>2624.8965</td>
          </tr>
          <tr>
            <td>Food and Beverages</td>
            <td>2673.5640</td>
          </tr>
        </table>
        
        <p><strong>Insights:</strong></p>
        
        <ol>
          <li><strong>Food and Beverages:</strong> With a gross income of $2673.5640, the Food and Beverages category 
          generates the highest revenue among all product lines.</li>
          <li><strong>Sports and Travel:</strong> Sports and Travel products follow closely behind with a gross income 
          of $2624.8965, indicating strong performance in generating revenue.</li>
          <li><strong>Electronic Accessories and Fashion Accessories:</strong> Both Electronic Accessories and Fashion 
          Accessories contribute significantly to the overall gross income, with $2587.5015 and $2585.9950 respectively.
          </li>
          <li><strong>Home and Lifestyle:</strong> Home and Lifestyle products generate a gross income of $2564.8530, 
          making a notable contribution to the company's revenue stream.</li>
          <li><strong>Health and Beauty:</strong> Health and Beauty products have a gross income of $2342.5590, 
          indicating their financial significance within the product lineup.</li>
        </ol>
        
        <p>These insights provide valuable information for assessing the financial performance of each product line and 
        can aid in strategic decision-making and resource allocation within the business.</p>
        '''
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Product Lines Gross Income">', explanation


def average_ratings_by_product_lines():
    mean_ratings = dataset.groupby('Product line')['Rating'].mean().reset_index()

    sns.barplot(x='Product line', y='Rating', data=mean_ratings)
    plt.xticks(rotation=45)
    plt.title('Average Ratings by Product Line')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
    <p><strong>Insights:</strong></p>

    <p>The table presents the average ratings for various product lines based on customer feedback:</p>
    
    <table>
      <tr>
        <th>Product Line</th>
        <th>Average Rating</th>
      </tr>
      <tr>
        <td>Electronic Accessories</td>
        <td>6.92</td>
      </tr>
      <tr>
        <td>Fashion Accessories</td>
        <td>7.03</td>
      </tr>
      <tr>
        <td>Food and Beverages</td>
        <td>7.11</td>
      </tr>
      <tr>
        <td>Health and Beauty</td>
        <td>7.00</td>
      </tr>
      <tr>
        <td>Home and Lifestyle</td>
        <td>6.84</td>
      </tr>
      <tr>
        <td>Sports and Travel</td>
        <td>6.92</td>
      </tr>
    </table>
    
    <p>Here are the key insights from the data:</p>
    
    <ol>
      <li><strong>Fashion Accessories:</strong> Fashion Accessories have the highest average rating of 7.03, indicating 
      strong customer satisfaction with this product category.</li>
      <li><strong>Food and Beverages:</strong> Food and Beverages closely follow with an average rating of 7.11, 
      suggesting high customer satisfaction and positive feedback for items in this category.</li>
      <li><strong>Health and Beauty:</strong> Health and Beauty products maintain a commendable average rating of 7.00, 
      reflecting positive customer sentiment towards these items.</li>
      <li><strong>Electronic Accessories and Sports and Travel:</strong> Both Electronic Accessories and Sports and 
      Travel products have an average rating of 6.92, indicating moderate to high satisfaction levels among customers.
      </li>
      <li><strong>Home and Lifestyle:</strong> Home and Lifestyle products have a slightly lower average rating of 6.84,
       suggesting room for improvement or potential areas where customer expectations may not be fully met.</li>
    </ol>
    
    <p>These insights provide valuable feedback for businesses to assess customer satisfaction, identify strengths and 
    weaknesses in product categories, and tailor strategies to enhance overall customer experience.</p>
    '''
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Average Ratings by Product Line">', explanation


def sales_volume_segmented_by_gender_plot():
    sns.barplot(x='Product line', y='Quantity', hue='Gender', data=dataset)

    plt.title('Sales Volume by Product Line, Segmented by Gender')
    plt.xticks(rotation=17)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    explanation = '''
    <p>Total profitability by product line provides a comprehensive insight into the financial performance of each 
    product category within a business. By examining the gross income generated by each product line, we gain valuable 
    understanding of their respective contributions to the overall profitability of the company.</p>

    <p>In the provided dataset, we observe the gross income for various product lines:</p>

    <ul>
        <li><strong>Electronic Accessories:</strong> The gross income for this product line stands at $2587.5015.</li>
        <li><strong>Fashion Accessories:</strong> This category yields a gross income of $2585.9950.</li>
        <li><strong>Food and Beverages:</strong> Generating $2673.5640 in gross income, this segment showcases a strong 
        performance.</li>
        <li><strong>Health and Beauty:</strong> Gross income in this sector amounts to $2342.5590, indicating its 
        financial significance.</li>
        <li><strong>Home and Lifestyle:</strong> With a gross income of $2564.8530, this category makes a notable 
        contribution to the company's profitability.</li>
        <li><strong>Sports and Travel:</strong> This product line shows a gross income of $2624.8965, highlighting its 
        competitive performance.</li>
    </ul>

    <p>Analyzing these figures allows us to discern the relative profitability of each product line. By identifying 
    high-performing sectors, businesses can allocate resources effectively, capitalize on strengths, and address 
    weaknesses. Moreover, this analysis facilitates strategic decision-making, enabling companies to optimize their 
    product mix, marketing strategies, and operational activities to enhance overall profitability.</p>
    '''

    return (f'<img src="data:image/png;base64,{plot_base64}" alt="Sales Volume by Product Line, Segmented '
            f'by Gender">'), explanation


def generate_monthly_income_plot():
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Month'] = dataset['Date'].dt.to_period('M')
    monthly_income = dataset.groupby(['Month', 'Product line'])['gross income'].sum().unstack()

    plt.figure(figsize=(10, 6))
    monthly_income.plot(kind='line', marker='o')
    plt.title('Monthly Gross Income by Product Line')
    plt.xlabel('Month')
    plt.ylabel('Gross Income')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
    <p>The provided data represents the monthly gross income for each product line over a period of time. Here's a 
    glimpse of the trends for the first three months of 2019:</p>

    <table>
      <tr>
        <th>Month</th>
        <th>Electronic Accessories</th>
        <th>Sports and Travel</th>
      </tr>
      <tr>
        <td>2019-01</td>
        <td>$896.7280</td>
        <td>$1031.7630</td>
      </tr>
      <tr>
        <td>2019-02</td>
        <td>$826.8050</td>
        <td>$657.6005</td>
      </tr>
      <tr>
        <td>2019-03</td>
        <td>$863.9685</td>
        <td>$935.5330</td>
      </tr>
    </table>

    <p>From this snippet, it appears that in January 2019, Sports and Travel products generated the highest gross 
    income, followed by Electronic Accessories. However, in February and March, Electronic Accessories saw a decrease 
    in gross income, while Sports and Travel maintained a relatively stable income level.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Monthly Gross Income by Product Line">', explanation


def generate_gross_income_by_gender_plot():
    plt.figure(figsize=(12, 6))
    gross_gender = sns.barplot(x='Product line', y='Quantity', hue='Gender', data=dataset)
    gross_gender.plot()
    plt.title('Gross Income by Product Line, Grouped by Gender')
    plt.xlabel('Product Line')
    plt.ylabel('Gross Income')
    plt.xticks(rotation=90)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
        <p>This plot shows the gross income for each product line, categorized by gender.<p>
        <p>It provides insights into how different product lines perform in terms of gross income 
        for both male and female customers.<p>
        '''

    return (f'<img src="data:image/png;base64,{plot_base64}" alt="Gross Income by Product Line, Grouped by Gender">',
            explanation)


def generate_monthly_gross_income_plot():
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Month'] = dataset['Date'].dt.to_period('M')
    monthly_income = dataset.groupby('Month')['gross income'].sum()

    plt.figure(figsize=(10, 6))
    monthly_income.plot(kind='line', marker='o')
    plt.title('Monthly Gross Income')
    plt.xlabel('Month')
    plt.ylabel('Gross Income')
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()

    plt.close()

    explanation = '''
    <p>The provided data illustrates the trend of monthly gross income over the first three months of 2019:</p>

    <table>
      <tr>
        <th>Month</th>
        <th>Gross Income</th>
      </tr>
      <tr>
        <td>2019-01</td>
        <td>$5537.708</td>
      </tr>
      <tr>
        <td>2019-02</td>
        <td>$4629.494</td>
      </tr>
      <tr>
        <td>2019-03</td>
        <td>$5212.167</td>
      </tr>
    </table>

    <p>This data suggests that there was a peak in gross income in January 2019, followed by a decrease in February, 
    and then a slight increase again in March. Understanding the fluctuations in monthly gross income can help 
    businesses in financial planning, identifying seasonal trends, and adjusting their strategies to maximize revenue 
    and profitability throughout the year.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Monthly Gross Income">', explanation


def generate_total_gross_income_by_branch_plot():
    branch_income = dataset.groupby('Branch')['gross income'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Branch', y='gross income', data=branch_income)
    plt.title('Total Gross Income by Branch')
    plt.xlabel('Branch')
    plt.ylabel('Total Gross Income')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()

    plt.close()

    explanation = '''
    <p>The provided data presents the total gross income for each branch:</p>

    <table>
      <tr>
        <th>Branch</th>
        <th>Gross Income</th>
      </tr>
      <tr>
        <td>A</td>
        <td>$5057.1605</td>
      </tr>
      <tr>
        <td>B</td>
        <td>$5057.0320</td>
      </tr>
      <tr>
        <td>C</td>
        <td>$5265.1765</td>
      </tr>
    </table>

    <p>From this information, it appears that Branch C has the highest total gross income, followed closely by Branch A 
    and then Branch B. Understanding the gross income distribution across branches can help businesses in evaluating 
    branch performance, allocating resources effectively, and identifying areas for improvement.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Total Gross Income by Branch">', explanation


def generate_profitability_plot():
    product_profitability = dataset.groupby('Product line')['gross income'].sum()

    plt.figure(figsize=(12, 6))
    product_profitability.plot(kind='bar', color='lightgreen')
    plt.title('Total Profitability by Product Line')
    plt.xlabel('Product line')
    plt.ylabel('Total Profitability')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    explanation = '''
    <p>Total profitability by product line provides a comprehensive insight into the financial performance of each 
    product category within a business. By examining the gross income generated by each product line, we gain valuable 
    understanding of their respective contributions to the overall profitability of the company.</p>

    <p>In the provided dataset, we observe the gross income for various product lines:</p>

    <ul>
        <li><strong>Electronic Accessories:</strong> The gross income for this product line stands at $2587.5015.</li>
        <li><strong>Fashion Accessories:</strong> This category yields a gross income of $2585.9950.</li>
        <li><strong>Food and Beverages:</strong> Generating $2673.5640 in gross income, this segment showcases a strong 
        performance.</li>
        <li><strong>Health and Beauty:</strong> Gross income in this sector amounts to $2342.5590, indicating its 
        financial significance.</li>
        <li><strong>Home and Lifestyle:</strong> With a gross income of $2564.8530, this category makes a notable 
        contribution to the company's profitability.</li>
        <li><strong>Sports and Travel:</strong> This product line shows a gross income of $2624.8965, highlighting its 
        competitive performance.</li>
    </ul>

    <p>Analyzing these figures allows us to discern the relative profitability of each product line. By identifying 
    high-performing sectors, businesses can allocate resources effectively, capitalize on strengths, and address 
    weaknesses. Moreover, this analysis facilitates strategic decision-making, enabling companies to optimize their 
    product mix, marketing strategies, and operational activities to enhance overall profitability.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Total Profitability by Product Line">', explanation


def generate_revenue_plot():
    product_revenue = dataset.groupby('Product line')['Total'].sum()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    product_revenue.plot(kind='bar', color='skyblue')
    plt.title('Total Revenue by Product Line')
    plt.xlabel('Product line')
    plt.ylabel('Total Revenue')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    explanation = '''
    <p>This metric provides valuable insights into the revenue-generating potential of different product lines, aiding 
    in strategic decision-making and resource allocation.</p>
    <p>In the provided dataset, we observe the total revenue for various product lines:</p>
    <ul>
        <li><strong>Electronic Accessories:</strong> This product line contributes a total revenue of $54,337.5315.</li>
        <li><strong>Fashion Accessories:</strong> Generating $54,305.8950 in total revenue, this category demonstrates 
        strong performance.</li>
        <li><strong>Food and Beverages:</strong> With a total revenue of $56,144.8440, this segment emerges as a 
        significant revenue generator.</li>
        <li><strong>Health and Beauty:</strong> The total revenue for this category amounts to $49,193.7390.</li>
        <li><strong>Home and Lifestyle:</strong> This product line yields a total revenue of $53,861.9130, indicating 
        its importance in the company's revenue stream.</li>
        <li><strong>Sports and Travel:</strong> Generating $55,122.8265 in total revenue, this category showcases robust
         performance.</li>
    </ul>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Total Revenue by Product Line">', explanation


def generate_sales_volume_plot():
    sales_volume = dataset.groupby('Product line')['Quantity'].sum()

    # Visualize sales volume
    plt.figure(figsize=(10, 6))
    sales_volume.sort_values().plot(kind='barh', color='salmon')
    plt.yticks(rotation=45)
    plt.title('Total Sales Volume by Product Line')
    plt.xlabel('Total Sales Volume')
    plt.ylabel('Product line')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    explanation = '''
    <p>From the provided data, it's clear that electronic accessories have the highest total sales volume among all 
    product lines, with 971 units sold. Following closely behind are sports and travel products, with a total sales 
    volume of 920 units. Fashion accessories and home and lifestyle items also perform strongly, with 902 and 911 units 
    sold respectively. Food and beverages, health and beauty products fall slightly behind in terms of total sales 
    volume, with 952 and 854 units sold respectively.</p>

    <p>This breakdown of sales volume by product line can help in understanding consumer preferences and market trends. 
    It indicates which product categories are more popular or in higher demand compared to others. Businesses can 
    utilize this information to adjust their marketing strategies, inventory management, and product offerings to better
     meet customer needs and maximize sales.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Total Sales Volume by Product Line">', explanation


@app.route('/', methods=['GET', 'POST'])
def index():
    title = 'Supermarket Sales Analysis'
    plot_type = None
    plot = None
    explanation = None

    if request.method == 'POST':
        visualization_type = request.form['visualization_type']
        plot, explanation = generate_visualization(visualization_type)
        plot_type = visualization_type.replace('_', ' ').title()

    return render_template('index.html', title=title, plot_type=plot_type, plot=plot,
                           explanation=explanation)


def generate_product_distribution_plot():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataset, x='Product line')
    plt.xticks(rotation=17)
    plt.title('Distribution of Product Line')
    plt.xlabel('Product Line')
    plt.ylabel('Count')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    explanation = '''
    <p>This plot shows the distribution of product lines in the supermarket sales dataset. Each bar represents a product
     line, and the height of the bar indicates the count of occurrences of that product line in the dataset.</p>

    <p>Some trends observed in the distribution:</p>
    <ul>
        <li>'Fashion accessories' has the highest count, indicating that it is the most frequently sold product line, 
        followed by 'Food and beverages' and 'Electronic accessories'.</li>
        <li>'Sports and travel' and 'Home and lifestyle' categories also show significant counts, indicating significant
         popularity among customers.</li>
        <li>'Health and beauty' has the lowest count among the product lines.</li>
    </ul>

    <p>Overall, this visualization provides insights into the popularity and demand for different product lines in the 
    supermarket.</p>
    '''

    return f'<img src="data:image/png;base64,{plot_base64}" alt="Distribution of Product Line">', explanation


@app.route('/predict_sales', methods=['GET', 'POST'])
def predict_sales():
    if request.method == 'POST':
        unit_price = float(request.form['unit_price'])
        quantity = int(request.form['quantity'])
        tax_percent = float(request.form['tax_percent'])
        gross_income = float(request.form['gross_income'])
        X = dataset[['Unit price', 'Quantity', 'Tax 5%', 'gross income']]

        y = dataset['Total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict([[unit_price, quantity, tax_percent, gross_income]])
        mse = mean_squared_error(y_test, model.predict(X_test))

        return render_template('predict_sales.html', prediction=prediction, unit_price=unit_price,
                               quantity=quantity, mse=mse, tax_percent=tax_percent, gross_income=gross_income)

    return render_template('predict_sales.html')


# Route to display the dataset
@app.route('/view_dataset')
def view_dataset():
    # Render the dataset.html template with the dataset
    return render_template('dataset.html', dataset=dataset_html)


if __name__ == '__main__':
    app.run(debug=True)
