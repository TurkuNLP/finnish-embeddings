import os
import zipfile
import gzip
import tempfile
import json
import rapidfuzz
import re
from collections import Counter
from tqdm import tqdm
import trafilatura


def get_all_html_files_by_date(date:str, path_to_html_files:str):
    return sorted([item.path for item in os.scandir(path_to_html_files) if item.name.startswith(date)])

def get_content_from_html(index:int, html_paths:list, json_title:str, score_cutoff=80, focus="balanced"):
    """Based on the index of the failed JSON object, match the titles of the JSON and HTML objects, and if successful, extract content from a corresponding HTML file.

    Parameters
    ----------
    index : int
        The index to use for mapping the JSON object with the corresponding HTML object.
    html_paths : list
        A list of filenames from a given date.
    json_title : str
        A title extracted from JSON metadata to be mapped with the possibly correpsonding HTML object.
    score_cutoff : int
        The value for determining if the two titles are similar enough to be belonging to the same article. Accepted range 0-100, where 100
            stands for exact match.
    focus : str ("balanced" | "precision" | "recall")
        If the Trafilatura HTML extraction should take a balanced approach, or favor precision or recall.

    Returns
    -------
    extracted_text : str
        The text extracted from HTML using the settings given in {focus}.
    -20 : int
        if the title similarity is below the {cutoff} value.
    -30 : int
        if not a pre-defined HTML tag ("yle__article__content") cannot be found.
    """

    with gzip.open(html_paths[index], "rt", encoding="utf-8") as file:
        html_content = file.read()
        extracted_title = trafilatura.extract_metadata(html_content).as_dict()["title"]
        if not rapidfuzz.fuzz.ratio(json_title, extracted_title, score_cutoff=score_cutoff): # returns 0.0 (evaluated as False), if the score is less than the cutoff value
            return -20
        if not "yle__article__content" in html_content:
            return -30
        # Only extract text if the titles are similar enough and a content tag can be found in HTML
        if focus == "precision":
            extracted_text = trafilatura.extract(html_content, favor_precision=True) # try to avoid parsing extra content with 'favor_precison'
        elif focus == "recall":
            extracted_text = trafilatura.extract(html_content, favor_recall=True) # try to avoid omitting essential content with 'favor_recall'
        else:
           extracted_text = trafilatura.extract(html_content) # use an approach that tries to balance precision and recall
        return extracted_text

def extract_data(obj:dict, index:int, html_paths:list|None, recovery_option:bool, extraction_focus="balanced"):
    """Extracts the wanted data (title, tags, summary, text (content), url, and timestamp) from a JSON object.
    If the 'content' key is not found in the object, tries to find the corresponding HTML file and extract the data from it.
    
    Returns a tuple where the first item gives information about the execution, and the second is the refined object, or None if extraction has failed.
    Informational return values:
    - 0: extraction successful from the JSON object
    - 1: content extraction successful from the HTML file
    - -1: the HTML couldn't be mapped to the JSON object because of a title mismatch
    - -2: the HTML didn't contain 'yle__article__content' tag(s)
    - -3: there was no recovery option (the number of HTML documents and JSON objects didn't match, or there were no HTML files)"""
    
    reason = 0

    # Check if there is 'tags' metadata available
    if "tags" not in obj:
        tags = []
    else:
        tags = [tag["term"] for tag in obj["tags"]]
    
    # Check if the object contains the 'content' key
    if "content" not in obj:
        # If not, check if HTML fallback can be tried instead
        if recovery_option:
            text = get_content_from_html(index, html_paths, obj["title"], focus=extraction_focus)
            if isinstance(text, int):
                if text == -20: return -1, None
                else: return -2, None
            reason = 1 # record the successfull HTML extraction
        else:
            return -3, None
    else:
        text = trafilatura.extract(obj["content"][0]["value"], favor_precision=True) # try to extract text from content, try to avoid parsing extra content with 'favor_precison'

    if not text:
        text = ""

    # If the 'id' key is empty, try to use 'link' instead
    url = obj["id"]
    if not url:
        url = obj["link"] # some ids are empty string, in which case get the RSS-link
   
    title = obj["title"]
    summary = obj["summary"]
    timestamp = obj["date_isoformat"]

    return reason, {
                    "title": title,
                    "tags": tags,
                    "summary": summary,
                    "text": text,
                    "url": url,
                    "timestamp": timestamp
                    }

def parse_broken_json(opened_file, end_of_object_line="    },\n"):
    """Tries to recover most articles from a file that has encoding errors.
    The current implementation only works for a predefined file format where indentation has been used."""

    # Return to the start of the fle
    opened_file.seek(0)

    # Create a variable for collecting the successfully encoded articles
    articles = []

    # Initialize an empty string to collect a full object line by line
    current_object = ""

    for i, line in enumerate(opened_file):
        # Skip the '[' character on the first line
        if i == 0:
            continue
        # Add the line to the string collecting the current object
        if line != end_of_object_line:
            current_object += line

        # When a known end-of-object-line is reached, remove the comma for avoiding errors in parsing
        else:
            line = line.replace(",", "")
            current_object += line

            # Only try to decode the object when end-of-the-object-line has been reached,
            # if done succesfully, add the article to the list and empty the string
            try:
                articles.append(json.loads(current_object))
                current_object = ""

            # If decoding is not succesfull, continue iterating the file and try to recover other entires
            # (not sure if it would be better to terminate the try altogether here, though)
            except json.JSONDecodeError as e:
                print(repr(e))
                print(repr(current_object))
                current_object = ""
                continue
    try:
        return articles
    
    # If articles is not defined, the file contents were empty, in which case just return an empty list
    except UnboundLocalError as e:
        print(str(e))
        return []

def initialize_stats_dict():
    return {"title_mismatch": 0,
            "no_content_in_html": 0,
            "json_html_mapping_error": 0,
            "recovered": 0,
            "json_success": 0,
            "num_articles" : 0}

def process_json(filepath, write_to, path_to_html_files, extraction_focus):
    """Iterates over all articles in one file and writes them to the specified file.
    In cases where the JSON object doesn't contain the content, hmtl_path is used for extracting the content from the corresponding html file."""

    def get_date():
        filename = filepath.rsplit("/", 1)[-1]
        return filename.split(".")[1] # e.g. 'yle-fi.2024-01-01.json' -> '2024-01-01'

    date = get_date()
    html_filepaths = get_all_html_files_by_date(date, path_to_html_files) if os.path.exists(path_to_html_files) else []
    error_stats = initialize_stats_dict()
    reason_mapping = {1: "recovered",
                      0: "json_success",
                      -1: "title_mismatch",
                      -2: "no_content_in_html",
                      -3: "json_html_mapping_error",
                      }

    with open(filepath) as json_file:
        try:
            data = json.load(json_file)
        except json.JSONDecodeError as json_error:
            data = parse_broken_json(json_file)
        
        assert isinstance(data, list), f"The structure of the JSON file at {filepath} doesn't match the expected format (a list)"
        num_articles = len(data)
        option_to_recover_from_html = True if len(html_filepaths) == num_articles else False # if indices can (be tried to) be naively mapped
        error_stats["num_articles"] = num_articles

        # Iterate over articles in the json file
        for i, article in enumerate(data):
            extraction_status, refined_data = extract_data(article, i, html_filepaths, option_to_recover_from_html, extraction_focus)
            reason = reason_mapping[extraction_status]
            error_stats[reason] += 1
            
            # Write data to file if the extraction was successful
            if refined_data:
                print(json.dumps(refined_data, ensure_ascii=False), file=write_to)
        
    return error_stats

def get_zip_filepaths(zip_dir):
    """Returns the file paths to the zip files as a sorted list."""
    return sorted(
        path for f in os.listdir(zip_dir) if zipfile.is_zipfile(path := os.path.join(zip_dir, f))
    )

def get_date(filename, date_len):
    """Extracts date from the HTML file names, where the filename is expected to start with the date."""
    date = filename[:date_len] # e.g. '2024-01-22-00-00-04---832a2c87751edb2bb30a61446387dad6.html' -> '2024-01-22'
    assert re.match(r"\d{4}-\d{2}-\d{2}", date), f"Error in date extraction: {date}"
    return date

def get_daily_counts(dir_path, date_format="yyyy-mm-dd"):
    date_len = len(date_format)
    dates = (get_date(file.name, date_len) for file in os.scandir(dir_path))
    return Counter(dates)

def combine_counters(json_counter, html_counter):
    """Combines the keys of two Counters (or counter-like dicts)."""
    all_dates = set(json_counter.keys()).union(set(html_counter.keys()))
    combined_counts = {}
    for date in all_dates:
        combined_counts[date] = {
            "json": json_counter.get(date, initialize_stats_dict()),
            "html": html_counter.get(date, 0)
        }
    return combined_counts

def rename(filepath:str):
    """Check if a filepath already exists, and if yes, return a new string from the given one
    by adding a trailing number enclosed in parenthesis (starting from '(1)'),
    or by incrementing an existing trailing number in the string.
    Will modify the last filepath component (filename).
    
    For example, rename('file.txt') would return 'file(1).txt'"""

    head, tail = os.path.split(filepath)
    name, extension = os.path.splitext(tail)
    end_pattern = r"\(\d{1,}\)"

    def increment_name(name):
        match = re.findall(end_pattern, name)
        if match:
            if name.endswith(match[-1]):
                num = int(match[-1][1:-1]) + 1 # remove parenthesis, increment by one
                name = name[:-len(match[-1])] + "(" + str(num) + ")"
        else:
            name = f"{name}(1)"
        return name
    
    new_filepath = os.path.join(head, f"{name}{extension}")

    while os.path.exists(new_filepath):
        name = increment_name(name)
        new_filepath = os.path.join(head, f"{name}{extension}")
    
    if tail != name+extension:
        print(f"Renamed {tail} to {name}{extension} to avoid overwriting")
    
    return new_filepath

def write_statistics_to_csv(combined_counts:dict, filename="article_count_stats.csv"):
    """Writes statistics about successful and failed parsing attempts and the number of available html articles.
    Expects the dictionary to have dates as keys and each value to be a dictionary in format
    {"json": {<initialize_stats_dict()>}, "html": <int>}"""
    
    while os.path.exists(filename):
        filename = rename(filename)

    header = "date,"+",".join(initialize_stats_dict().keys())+",html_total"

    with open(filename, "w") as file:
        print(header, file=file)
        for key in sorted(combined_counts.keys()):
            json_values_str = (str(i) for i in combined_counts[key]["json"].values())
            print(key, ",".join(json_values_str), combined_counts[key]["html"], sep=",", file=file)
            
def filter_news_data(zip_dir:str, local_scratch:str, destination_file:str, write_statistics:str=None, extraction_focus:str="balanced", overwrite:bool=True):
    """From a directory of zipped folders, iterates over each zip and temporarily extracts the contents to a given location.
    Iterates over news data contained in JSON files, and extracts data from HTML files whenever needed and possible.
    Writes it into a specified jsonl file. Optionally, writes statistics about the data extraction status.
    
    Parameters
    ----------
    zip_dir : str
        Path to the directory that contains the zip files, given as a string.
    local_scratch : str
        Path to the temporary storage, given as a string.
    destination_file : str
        Name for the resulting file.
    write_statistics : str
        Name for the file including the extraction statistics. If None, the files will not be written.
    extraction_focus : str
        Accepted values: "balanced", "precision", or "recall". Determines the focus of HTML text extraction.
    overwrite : bool
        If True, doesn't check if a file named {destination_file} already exists.
        If False, will rename {destination_file} if a file with same name already exists.
    """
    
    zip_filepaths = get_zip_filepaths(zip_dir)
    destination_path = os.path.join(local_scratch, destination_file)

    if write_statistics:
        stats_path = os.path.join(local_scratch, write_statistics)

    if not overwrite:
        destination_path = rename(destination_path)

        if write_statistics:
            stats_path = rename(stats_path)

    date_len = len("yyyy-mm-dd")

    # Open a file for writing the refined data in jsonl format
    with open(destination_path, "w") as dest:

        if write_statistics:
            stats = {}
            html_stats = {}

        # Iterate over each zip file
        for zip_path in tqdm(zip_filepaths, desc="Processing Zip Files per Year"):

            # Extract the zip file into a temporary directory for processing
            with tempfile.TemporaryDirectory(dir=local_scratch) as temp_dir:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                    # There should be one directory in temp_dir
                    paths = os.listdir(temp_dir)
                    assert len(paths) == 1, f"More than one item found in the extracted zip directory."
                    json_dir = os.path.join(temp_dir, paths[0])
                    assert os.path.isdir(json_dir), "The element in the extracted directory is not a directory."

                    # Iterate over json files in the extracted zip file
                    path_to_html_dir = os.path.join(temp_dir, json_dir, "yle-fi-downloads/") # needed in cases when an article object in the json file doesn't contain the actual article
                    with os.scandir(json_dir) as contents:
                        for item in sorted(list(contents), key=lambda x: x.name):
                            if item.is_file and item.name.endswith(".json"):
                                processing_info = process_json(item.path, dest, path_to_html_dir, extraction_focus)
                                if write_statistics:
                                    date = item.name.split(".")[1] # e.g. 'yle-fi.2024-01-01.json' -> '2024-01-01'
                                    assert len(date) == date_len, f"Date extraction failed: {date}"
                                    stats[date] = processing_info
                    if write_statistics:
                        if os.path.exists(path_to_html_dir):
                            html_stats.update(get_daily_counts(dir_path=path_to_html_dir))
                    
        if write_statistics:
            full_stats = combine_counters(stats, html_stats)
            write_statistics_to_csv(full_stats, filename=stats_path)


def main():

    LOCAL_SCRATCH = os.getenv("LOCAL_SCRATCH")
    if not LOCAL_SCRATCH:
        print("The path to temporary storage needs to be specified for extracting the files. Please specify the path and try again. Exiting the program.")
        exit(1)

    NEWS_DATA = os.getenv("NEWS_DATA")
    if not NEWS_DATA:
        print("The path to the zip files is not specified. Please specify the path and try again. Exiting the program.")
        exit(1)
    
    EXTRACTION_FOCUS = os.getenv("EXTRACTION_FOCUS", "precision")
    assert EXTRACTION_FOCUS in ("balanced", "precision", "recall"), f"'{EXTRACTION_FOCUS}' is not in supported focus types. Choose between 'balanced', 'precision', or 'recall'."
    
    DESTINATION_FILE = os.getenv("DESTINATION_FILE", f"full-yle-news-corpus-{EXTRACTION_FOCUS}.jsonl")
    STATS_FILE = os.getenv("STATS_FILE", f"full-yle-news-statistics-{EXTRACTION_FOCUS}.csv")
    
    filter_news_data(NEWS_DATA, LOCAL_SCRATCH, DESTINATION_FILE, STATS_FILE, EXTRACTION_FOCUS)


if __name__ == "__main__":
    main()