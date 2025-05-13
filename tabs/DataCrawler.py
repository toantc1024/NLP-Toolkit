import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import re
import random
import os

def app():
    # Initialize session state for dataframe if not exists
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Write title
    st.title('Web Content Crawler')

    # Site configuration section
    st.subheader('Configure Crawler')

    site_type = st.selectbox(
        'Select website type or custom',
        ['IMDB Reviews', 'Amazon Reviews', 'Generic Site (Custom CSS Selectors)']
    )

    # URL input
    url = st.text_input(
        'Enter the URL of the webpage you want to crawl:',
        'https://www.imdb.com/title/tt0111161' if site_type == 'IMDB Reviews' else ''
    )

    # Crawler settings based on site type
    if site_type == 'IMDB Reviews':
        # Pre-configured settings for IMDB
        item_selector = st.text_input('Item CSS Selector', 'div[data-testid="review-card-parent"]')
        title_selector = st.text_input('Title CSS Selector', '.ipc-title__text')
        content_selector = st.text_input('Content CSS Selector', '.ipc-html-content-inner-div')
        rating_selector = st.text_input('Rating CSS Selector', '.ipc-rating-star--rating')
        load_more_selector = st.text_input('Load More Button Selector (if any)', 'button[class*="ipc-see-more__button"]')
        
        # Show additional IMDB-specific options
        review_page_selector = st.text_input('Reviews Page Link Selector', 'div[data-testid="reviews-header"] a.ipc-title-link-wrapper')
        helpful_votes_selector = st.text_input('Helpful Votes Selector', '.ipc-voting__label__count.ipc-voting__label__count--up')
        unhelpful_votes_selector = st.text_input('Unhelpful Votes Selector', '.ipc-voting__label__count.ipc-voting__label__count--down')
        
    elif site_type == 'Amazon Reviews':
        # Pre-configured settings for Amazon
        item_selector = st.text_input('Item CSS Selector', 'div[data-hook="review"]')
        title_selector = st.text_input('Title CSS Selector', 'a[data-hook="review-title"]')
        content_selector = st.text_input('Content CSS Selector', 'span[data-hook="review-body"]')
        rating_selector = st.text_input('Rating CSS Selector', 'i[data-hook="review-star-rating"]')
        load_more_selector = st.text_input('Load More Button Selector (if any)', '.a-pagination .a-last a')
        
    else:
        # Custom selectors for other sites
        item_selector = st.text_input('Item CSS Selector (containing elements to scrape)', 'div.item')
        title_selector = st.text_input('Title CSS Selector', '.title')
        content_selector = st.text_input('Content CSS Selector', '.content')
        rating_selector = st.text_input('Rating CSS Selector (optional)', '.rating')
        load_more_selector = st.text_input('Load More Button Selector (if any)', 'button.load-more')

    # General crawling settings
    max_items = st.number_input('Max items to crawl', min_value=1, value=100)
    timeout_seconds = st.number_input('Page load timeout (seconds)', min_value=1, value=10)
    scroll_pause_time = st.number_input('Scroll pause time (seconds)', min_value=0.1, value=1.0, step=0.1)

    # Advanced options
    with st.expander("Advanced Options"):
        headless = st.checkbox('Run in headless mode (no browser UI)', value=True)
        wait_for_javascript = st.number_input('Wait for JavaScript (seconds)', min_value=0, value=4)
        scroll_to_load = st.checkbox('Scroll to load more content', value=True)
        max_retries = st.number_input('Max retries for failed operations', min_value=1, value=3)
        
        # Advanced PDF options
        st.subheader("PDF Loader Options")
        extract_images = st.checkbox('Extract images from PDFs', value=False, 
                                    help="Extracts and processes images found in PDFs")
        extraction_mode = st.selectbox('PDF Extraction Mode', 
                                      ['plain', 'layout'],
                                      help="'plain' for basic extraction, 'layout' preserves document layout")
        
        # More advanced options for bypassing website restrictions
        st.subheader("Anti-Detection Options")
        use_random_user_agent = st.checkbox('Use random user agent', value=True, 
                                        help="Changes browser identification to avoid detection")
        add_browser_arguments = st.checkbox('Add stealth browser arguments', value=True,
                                        help="Adds arguments to make the browser harder to detect")
        disable_automation_flags = st.checkbox('Disable automation flags', value=True,
                                            help="Makes the browser appear more like a regular user browser")
        add_delay = st.checkbox('Add random delays', value=True,
                            help="Adds small random delays to appear more human-like")
        
        # Direct browser control
        direct_browser = st.checkbox('Use direct browser control for complex sites', value=False,
                                    help="Opens a regular Chrome browser - may help with sites that block automated access")
        if direct_browser:
            st.warning("This will open an actual browser window. Make sure you have Chrome installed.")
            headless = False

    # Web crawler button
    submit_button = st.button('Start Web Crawling')

    if submit_button:
        # Setup Chrome options for headless mode
        chrome_options = Options()
        
        # Apply anti-detection measures if selected
        if use_random_user_agent:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36 Edg/92.0.902.84"
            ]
            chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        if headless and not direct_browser:
            chrome_options.add_argument("--headless")
        
        # Basic options for all modes
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Additional stealth options
        if add_browser_arguments:
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
        if disable_automation_flags:
            chrome_options.add_experimental_option("useAutomationExtension", False)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        with st.spinner('Setting up browser...'):
            driver = webdriver.Chrome(options=chrome_options)
            
            # Additional anti-detection measures after driver is created
            if disable_automation_flags:
                # Execute JavaScript to modify navigator properties
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                
            wait = WebDriverWait(driver, timeout_seconds)
            short_wait = WebDriverWait(driver, 2)

        st.write('Crawling in progress...')
        
        # Initialize data collection
        items_data = []
        
        try:
            # Open the target website
            driver.get(url)
            
            # Add random delay if configured
            if add_delay:
                time.sleep(wait_for_javascript + random.uniform(0.5, 2.0))
            else:
                time.sleep(wait_for_javascript)  # Wait for JS to load
            
            # Check for access issues
            if "403 Forbidden" in driver.title or "Access Denied" in driver.page_source:
                st.error("Access denied by the website. Consider using direct browser control option.")
                if not direct_browser:
                    st.info("Retrying with direct browser option...")
                    driver.quit()
                    
                    # Try again with direct browser
                    new_options = Options()
                    new_options.add_argument("--window-size=1920,1080")
                    new_options.add_argument("--disable-notifications")
                    new_options.add_argument("--disable-popup-blocking")
                    driver = webdriver.Chrome(options=new_options)
                    driver.get(url)
                    time.sleep(wait_for_javascript)
                    
                    # If still blocked, give up
                    if "403 Forbidden" in driver.title or "Access Denied" in driver.page_source:
                        st.error("Still blocked. Try changing URL or using a different browser.")
                        driver.quit()
                        st.stop()
            
            # Handle IMDB specific navigation to reviews page
            if site_type == 'IMDB Reviews':
                try:
                    # Look for the reviews page link
                    review_page_button = None
                    try:
                        review_page_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, review_page_selector)))
                    except TimeoutException:
                        # Try clicking on the Reviews tab instead
                        alternative_selectors = [
                            'a[href*="reviews"]', 
                            'li.ipc-tab a[href*="reviews"]',
                            'div[data-testid="reviews"] a',
                            'a.ipc-tab[href*="reviews"]'
                        ]
                        
                        for selector in alternative_selectors:
                            try:
                                review_page_button = driver.find_element(By.CSS_SELECTOR, selector)
                                break
                            except NoSuchElementException:
                                continue
                    
                    if review_page_button:
                        href = review_page_button.get_attribute('href')
                        st.info(f"Navigating to reviews page: {href}")
                        driver.get(href)
                        time.sleep(2)  # Wait longer for the reviews page to load
                    else:
                        # Check if we're already on the reviews page
                        if "reviews" in driver.current_url:
                            st.info("Already on the reviews page")
                        else:
                            st.warning("Could not find reviews link. Attempting to modify URL directly.")
                            # Try to construct the reviews URL directly
                            movie_id = re.search(r'/(tt\d+)/', url)
                            if movie_id:
                                reviews_url = f"https://www.imdb.com/title/{movie_id.group(1)}/reviews"
                                driver.get(reviews_url)
                                time.sleep(2)
                            else:
                                st.warning("Could not determine the movie ID from the URL.")
                except Exception as e:
                    st.warning(f"Error navigating to reviews page: {str(e)}. Continuing with current page.")
            
            # Extract page title for reference
            page_title = driver.title
            st.write(f'Crawling: {page_title}')
            
            # Load more content if available
            item_count = 0
            
            if scroll_to_load:
                # Scroll to load dynamic content
                last_height = driver.execute_script("return document.body.scrollHeight")
                scroll_attempts = 0
                max_scroll_attempts = 20  # Limit scrolling attempts
                
                with st.spinner("Loading more content by scrolling..."):
                    while item_count < max_items and scroll_attempts < max_scroll_attempts:
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        
                        # Add randomization to scrolling if selected
                        if add_delay:
                            time.sleep(scroll_pause_time + random.uniform(0.1, 0.5))
                        else:
                            time.sleep(scroll_pause_time)
                        
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        
                        # Try to click "Load More" button if exists
                        if load_more_selector:
                            try:
                                load_more = short_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, load_more_selector)))
                                driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
                                time.sleep(0.5)  # Give it a moment to be properly visible
                                driver.execute_script("arguments[0].click();", load_more)
                                time.sleep(scroll_pause_time * 1.5)  # Wait longer after clicking
                                st.info("Clicked 'Load More' button")
                            except:
                                pass  # No load more button found or not clickable
                        
                        # Count items after each scroll to see if we've reached the limit
                        item_elements = driver.find_elements(By.CSS_SELECTOR, item_selector)
                        new_item_count = len(item_elements)
                        
                        if new_item_count > item_count:
                            item_count = new_item_count
                            st.info(f"Found {item_count} items so far...")
                            scroll_attempts = 0  # Reset attempts counter when we find new items
                        else:
                            scroll_attempts += 1
                        
                        if new_height == last_height:
                            scroll_attempts += 1  # Count as an attempt if height doesn't change
                        
                        last_height = new_height
                        
                        if item_count >= max_items:
                            st.success(f"Reached maximum item count: {max_items}")
                            break
                    
                    if scroll_attempts >= max_scroll_attempts:
                        st.info("Stopped scrolling: no new content appearing")
            
            # Find all item elements
            item_elements = driver.find_elements(By.CSS_SELECTOR, item_selector)
            
            if not item_elements:
                st.warning(f"No items found with selector: {item_selector}")
                # Take screenshot for debugging
                screenshot_path = f"screenshot_{int(time.time())}.png"
                driver.save_screenshot(screenshot_path)
                st.warning(f"Screenshot saved to {screenshot_path} for debugging")
                
                # Try again with broader selector
                st.info("Trying with a broader selector...")
                
                # Site-specific fallback selectors
                if site_type == "IMDB Reviews":
                    fallback_selectors = [
                        ".review-container", 
                        ".lister-item",
                        "div.review"
                    ]
                elif site_type == "Amazon Reviews":
                    fallback_selectors = [
                        ".review", 
                        "[data-hook='review']",
                        ".a-section.review"
                    ]
                else:
                    fallback_selectors = [
                        "div.item", 
                        "article", 
                        ".card",
                        ".list-item"
                    ]
                    
                # Try each fallback selector
                for fallback_selector in fallback_selectors:
                    st.info(f"Trying with selector: {fallback_selector}")
                    item_elements = driver.find_elements(By.CSS_SELECTOR, fallback_selector)
                    if item_elements:
                        st.success(f"Found {len(item_elements)} items with fallback selector: {fallback_selector}")
                        item_selector = fallback_selector
                        break
            
            # Initialize progress bar
            progress_bar = st.progress(0)
            
            # Limit to max_items
            item_elements = item_elements[:max_items]
            
            # Check if we actually found items
            if not item_elements:
                st.error("Could not find any items with the provided selectors.")
                # Take screenshot for debugging if not already taken
                screenshot_path = f"screenshot_{int(time.time())}.png"
                driver.save_screenshot(screenshot_path)
                st.info(f"Page screenshot saved to {screenshot_path} for debugging")
                
                # Show page source for debugging
                with st.expander("Page Source (for debugging)"):
                    st.code(driver.page_source[:5000] + "...", language="html")
                    
                driver.quit()
                st.stop()
            
            # Process each item
            for i, item_element in enumerate(item_elements):
                try:
                    item_data = {}
                    
                    # Extract title
                    try:
                        title_element = item_element.find_element(By.CSS_SELECTOR, title_selector)
                        item_data['title'] = title_element.text
                    except:
                        item_data['title'] = "N/A"
                    
                    # Extract content
                    try:
                        content_element = item_element.find_element(By.CSS_SELECTOR, content_selector)
                        item_data['content'] = content_element.text
                    except:
                        item_data['content'] = "N/A"
                    
                    # Extract rating if available
                    try:
                        rating_element = item_element.find_element(By.CSS_SELECTOR, rating_selector)
                        rating_text = rating_element.text
                        
                        # Extract numeric rating if possible
                        if site_type == 'IMDB Reviews':
                            try:
                                current_rating = item_element.find_element(By.CSS_SELECTOR, '.ipc-rating-star--rating').text
                                max_rating = item_element.find_element(By.CSS_SELECTOR, '.ipc-rating-star--maxRating').text
                                item_data['rating'] = f'{current_rating}/{max_rating}'
                            except:
                                item_data['rating'] = rating_text
                        else:
                            # Try to extract numeric rating from text with regex
                            rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
                            if rating_match:
                                item_data['rating'] = rating_match.group(1)
                            else:
                                item_data['rating'] = rating_text
                    except:
                        item_data['rating'] = "N/A"
                    
                    # For IMDB, extract votes
                    if site_type == 'IMDB Reviews':
                        try:
                            helpful_votes = item_element.find_element(By.CSS_SELECTOR, helpful_votes_selector).text
                            item_data['helpful_votes'] = int(helpful_votes) if helpful_votes.isdigit() else 0
                            
                            unhelpful_votes = item_element.find_element(By.CSS_SELECTOR, unhelpful_votes_selector).text
                            item_data['unhelpful_votes'] = int(unhelpful_votes) if unhelpful_votes.isdigit() else 0
                        except:
                            item_data['helpful_votes'] = 0
                            item_data['unhelpful_votes'] = 0
                    
                    # Add the item data to our collection
                    items_data.append(item_data)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(item_elements))
                    
                except Exception as e:
                    st.warning(f"Error extracting data from item {i+1}: {str(e)}")
                    continue
                    
            # Create DataFrame from collected data
            if items_data:
                df = pd.DataFrame(items_data)
                st.session_state.df = df  # Store in session state for later use
                st.success(f"Successfully crawled {len(items_data)} items!")
                
                # Display the dataframe
                st.subheader("Crawled Data")
                st.write(df)
                
                # Column renaming section
                st.subheader("Customize Column Names Before Download")
                
                rename_columns = {}
                for col in df.columns:
                    new_col_name = st.text_input(f"Rename '{col}' to:", col)
                    if new_col_name != col:
                        rename_columns[col] = new_col_name
                
                # Apply column renaming if needed
                if rename_columns:
                    df_renamed = df.rename(columns=rename_columns)
                    st.write("Preview with renamed columns:")
                    st.write(df_renamed)
                    download_df = df_renamed
                else:
                    download_df = df
                
                # Export function
                def export_to_csv(df):
                    try: 
                        return df.to_csv(index=False).encode('utf-8')
                    except Exception as e:
                        st.error(f"Error exporting to CSV: {str(e)}")
                        return None
                
                # Download button
                csv_data = export_to_csv(download_df)
                if csv_data:
                    st.download_button(
                        label='Download as CSV',
                        data=csv_data,
                        file_name=f'crawled_data_{int(time.time())}.csv',
                        mime='text/csv'
                    )
                    
            else:
                st.error("No items were successfully crawled. Try adjusting the selectors.")
                    
        except Exception as e:
            st.error(f"Error during crawling: {str(e)}")
        finally:
            driver.quit()
            
    # Display existing data if available
    elif st.session_state.df is not None:
        st.subheader("Previously Crawled Data")
        st.write(st.session_state.df)
        
        # Column renaming section for existing data
        st.subheader("Customize Column Names Before Download")
        
        rename_columns = {}
        for col in st.session_state.df.columns:
            new_col_name = st.text_input(f"Rename '{col}' to:", col)
            if new_col_name != col:
                rename_columns[col] = new_col_name
        
        # Apply column renaming if needed
        if rename_columns:
            df_renamed = st.session_state.df.rename(columns=rename_columns)
            st.write("Preview with renamed columns:")
            st.write(df_renamed)
            download_df = df_renamed
        else:
            download_df = st.session_state.df
        
        # Download button for existing data
        def export_to_csv(df):
            try: 
                return df.to_csv(index=False).encode('utf-8')
            except Exception as e:
                st.error(f"Error exporting to CSV: {str(e)}")
                return None
        
        csv_data = export_to_csv(download_df)
        if csv_data:
            st.download_button(
                label='Download as CSV',
                data=csv_data,
                file_name=f'crawled_data_{int(time.time())}.csv',
                mime='text/csv'
            )
            
        # Send to pipeline
        if st.button("Send to Data Pipeline"):
            if 'pipeline_df' not in st.session_state:
                st.session_state.pipeline_df = pd.DataFrame()
            st.session_state.pipeline_df = download_df
            if 'processing_log' not in st.session_state:
                st.session_state.processing_log = []
            st.session_state.processing_log.append(f"Imported {len(download_df)} items from crawler")
            st.success("Data saved to pipeline!")
            
            # Navigation buttons with fixed paths
            if st.button("Go to Data Pipeline", key="existing_goto_pipeline"):
                st.session_state.active_tab = "Data Pipeline"
                st.rerun()

    # Navigation at the bottom                
    st.divider()
    if 'pipeline_df' in st.session_state:
        if st.button("Return to Pipeline"):
            st.session_state.active_tab = "Data Pipeline"
            st.rerun()
