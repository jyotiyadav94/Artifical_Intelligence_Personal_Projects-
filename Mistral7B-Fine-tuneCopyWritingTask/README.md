# Fine-Tuning LLM Model for Copywriting Agent

We are  fine-tuning a Language Model (LLM) for various copywriting tasks, using different datasets and prompts. The tasks include generating restaurant menu descriptions, social media posts, advertising copy, and newsletter campaigns.

For Demo HuggingFace [Demo](https://huggingface.co/spaces/Jyotiyadav/AzzuCopyWriterAgent)

For Deployment Notebook [Deployment](https://github.com/jyotiyadav94/ProductionReady)

For Dataset preparation [Dataset Preparation](https://github.com/jyotiyadav94/datasetnotebooks)

Table of Contents

- Setup
- Datasets
- Use Cases 
    * **Use Case 1**: Restaurant Menu Descriptions
    * **Use Case 2**: Social Media Posts
    * **Use Case 3**: Advertising Copy
    * **Use Case 4**: Newsletter Campaigns

## Use Case 1: Restaurant Menu Descriptions

Input:  Name of the dish.

Output:

Ingredients
Description
Allergens (from a predefined list)
Additional information (from a predefined list)

Prompt:

``` bash
f"""
As a restaurant menu manager, your role is to gather below informations based on input data {inputData} (Name of the dish).
generate the output

### information to be extracted :
<Ingredients>: Only Ingredients included in the dish.
<Description>: Briefly describe the dish.
<Allergens>: Only Choose relevant options from this list - [Cereals, Crustaceans, Egg, Fish, Peanuts, SOYBEAN, Latte, Nuts, Celery, Mustard, Sesame seeds, Sulfur dioxide and sulphites, Shell, Clams].
<Additional Information>: Only Choose relevant options from this list - [Spicy, Vegan, Gluten free, Vegetarian].

### Output Format
{{
ingredients: All Ingredients in a List,
description: Description in a string,
allergen: All allergen in a List,
Additional_information: All Additional_information in a List
}}

### Input data:
{inputData}

### Output:
"""
```

## Use Case 2: Social Media Posts
Input:

Can belong to any of these categories: Notoriety, Promotions, Reservations, Delivery, Theme Days, Menu.

Output:

Four social media posts for Facebook and Instagram.

Prompt:
``` bash
f"""
As the social media manager of the restaurant, your task is to craft four Social Media Posts for Restaurant for their Facebook and Instagram pages based on the Input data [which contains a goal for the posts and Image description which is an additional information for helping to write a posts].

### Guidelines:
* Craft the post with the goal of highlighting Input data.
* Incorporate 3-5 emojis, ensuring no more than one emoji is used every two sentences.
* Mention a maximum of 1 or 2 products from the menu.
* Remember that social media posts are part of a content plan, not sponsored content.
* Focus on showcasing the restaurant's strengths rather than directly promoting sales.
* Include a Call to Action mentioning information such as opening hours, restaurant address, telephone number, or WhatsApp number, if available.
* Direct audience attention to the online menu available at www.restaurants.menu.
* Utilize hashtags at the end of the description, relevant to the content and objectives. Use the # symbol to add hashtags.

### Information to be Extracted:
Generate four posts based on the above guidelines.

### Output Format:
{{
"Post1": "This is the content of Post 1.",
"Post2": "This is the content of Post 2.",
"Post3": "This is the content of Post 3.",
"Post4": "This is the content of Post 4."
}}

### Input data:
{inputData}{image}

### Output:
"""
```

Use Case 3: Advertising Copy

Input:

Can belong to any of these themes: Ideal Customer, Marriage, Anniversary, Birthdays, etc.
Interests of the people (e.g., Age: 18-69, Facebook Interest: Restaurants).

Output:

Four ad descriptions.

Prompt:

``` bash
f"""
As the advertising manager of the restaurant, your task is to create compelling ad copy for a restaurant's based on the Input data [which contains a goal for the descriptions and Buyers Personas which is an additional information for helping to write a descriptions].
The restaurant aim to attract a specific target audience described as the Buyers Personas.

### Guidelines:
* The Call to Action should target the Buyer Personas.
* Use emojis in the ad copy only if necessary.
* Mention 2 products from menu.
* Generated ad copy should be in only English language.
* Create content related hashtags at the end of the ad copy.
* Do not use restaurant info directly, use inside the generated ad text.

### Information to be Extracted:
Generate four descriptions based on the above guidelines.

### Output Format:
{{
"Description1": "This is the content of Description 1.",
"Description2": "This is the content of Description 2.",
"Description3": "This is the content of Description 3.",
"Description4": "This is the content of Description 4."
}}

### Input data:
{inputData}{buyersPersonas}
### Output:
"""
```


## Use Case 4: Newsletter Campaigns
Input:

Any Topic (Promotions, offers etc)

Output:

Campaign Name
Campaign Objective
Campaign Email

Prompt:
``` bash
f"""
As a Newsletter Manager, your task is to extract informations based on the input data {inputData}.
### information to be extracted :
<campaign Name>: Identifies a marketing initiative
<campaign Object>: Defines the primary goal of a marketing campaign
<campaign Email>: Communication sent via email as part of a marketing campaign.

### Output Format
{{
"campaignName": [Suggest some good campaign Name]
"campaignObject": [Suggest some good campaign Object],
"campaignEmail": [Write a sample campaign Email based on Campaign Name and campaign Object],
}}

### Input data:
{inputData}
### Output:
"""
```
