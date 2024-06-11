## **Project Structure**

``` bash
fastApi/
│
├── copywritingAgent/
│ ├── __init__.py
│ ├── router.py
│ ├── main.py
│ ├── .env
│
├── Dockerfile
├── README.md
├── docker-compose.yaml
└── requirements.txt
```


## Copywriting Agent API

This API provides endpoints for interacting with a copywriting agent. It allows users to generate copy for various purposes such as menu items, social media posts, advertising content, and newsletters.


## **How to Run**

1. **Download or Clone Repository:**
   - Download or clone the repository branch using the method of your choice. For example, if you have git installed on your machine, run the following command:
     ```bash
     https://github.com/jyotiyadav24/productionReady.git
     ```
     This will download the files into the fastApi folder.

2. **Ensure Python and Docker Installed:**
   - Ensure that both Python and Docker are installed on your machine.

3. **Navigate to the fastApi folder:**
   - Open a terminal and navigate to the fastApi folder:
     ```bash
     cd fastApi
     ```
   - Create a file named `.env` in the copywritingAgent folder and add the following credentials:
     
4. **Run Docker Images:**
   - Run the following command to start the Docker containers:
     ```bash
     docker-compose up
     ```
6. **Ensure Port Mapping:**
   - Ensure that the port mapping for the web application inside docker-compose.yml is set to port 8000, which is the default port.

7. **Test the Installation:**
   - Once the containers are running, you can test if the installation was successful by visiting the endpoints diiferent Endpoints below
     ```bash
     http://0.0.0.0:8000/docs/
     ```
8. **Stop Docker Containers:**
   - Open a new terminal and run the following command to stop the Docker containers:
     ```bash
     docker-compose down
     ```


## Endpoints
#### Process Menu
- **Endpoint:** `/menu/`
- **Method:** POST
- **Description:** Generates copy for menu items based on the provided goal.
- **Request Body:**
  - `goal`: str - It should be the name of the Dish. For eg: Pizza margherita,Arancini etc.
- **Response:**
  - Returns the generated copy for the menu item.
 ![Alt text](<images/menuWebpage1.png>)
  ![Alt text](<images/menujsonOutput.png>)
 ![Alt text](<images/menu.png>)
    

#### Process Social Media
- **Endpoint:** `/socialMedia/`
- **Method:** POST
- **Description:** Generates copy for social media posts along with an optional image upload.
- **Request Body:**
  - `goal`: str - The goal or purpose for generating the social media post copy.Goal can be from any of them ['Notoriety','Promotions','Reservations','Delivery','Theme days','Menu']
  - `image`: file - Optional. Image file to be included in the social media post.
- **Response:**
  - Returns the generated copy for the social media post.
  ![Alt text](<images/social2.png>)
  ![Alt text](<images/social1.png>)
 ![Alt text](<images/social.png>)


#### Process Advertising
- **Endpoint:** `/advertising/`
- **Method:** POST
- **Description:** Generates copy for advertising content based on the provided goal and target interest.
- **Request Body:**
  - `goal`: str - The goal or purpose for generating the advertising content.Goal ['Ideal customer','Returning From Holidays' etc ]
  - `interest`: str - The target interest or audience for the advertising content.['Age:18-65 Behaviors:Returned from a trip 2 weeks ago or 1 week ago']
- **Response:**
  - Returns the generated copy for the advertising content.
  ![Alt text](<images/advertising1.png>)
  ![Alt text](<images/Advertisingdesc.png>)
 ![Alt text](<images/advertising.png>)


#### Process Newsletter
- **Endpoint:** `/newsletter/`
- **Method:** POST
- **Description:** Generates copy for newsletter content based on the provided goal.
- **Request Body:**
  - `goal`: str - The goal or purpose for generating the newsletter content.['Notoriety','Events','Promotions, Offers','Holidays, Theme days','Delivery','Reservations','Thanks, Feedback']
- **Response:**
  - Returns the generated copy for the newsletter content.
![Alt text](<images/Newsletter1.png>)
![Alt text](<images/NewsletterOutput.png>)
 ![Alt text](<images/newsletter.png>)


#### .env


#### Error Handling
- If an error occurs during the processing of any request, the API will respond with an appropriate HTTP status code along with a detailed error message.

### .html Page In Progress 

### Fine-Tuned Model In Progress 
In general to check the Fine-tuned model size
print(model.get_memory_footprint())

```python

   ## Comment this code -----------------------
    llm = ChatGroq(temperature=temp,
                   groq_api_key=GROQ_API_KEY3,
                   model_name="mixtral-8x7b-32768",
                   max_tokens=max_tokens,
                   top_p=top_p,
                   frequency_penalty=frequency_penalty,
                   presence_penalty=presence_penalty)
    ## Comment this code -----------------------
```

Add Use the below code The below model requires GPU. 
use_auth_token=huggingface_token

```python

    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,)
    
    device = get_device_map()  # 'cpu'
    
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=huggingface_token)
    
    model =AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16,use_auth_token=huggingface_token)
    print(model.get_memory_footprint())
    print(model.get_memory_footprint())
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        temperature=temp,
        top_p=top_p,
        return_full_text=True,
        task='text-generation',
        max_new_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        num_return_sequences=1,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(pipeline=generate_text)
```
