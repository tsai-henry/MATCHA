import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--desc", help="Path to incident description text file")
parser.add_argument("--involved", help="Path to involved parties text file")
parser.add_argument("--syllabus", help="Path to extra info / syllabus text file")
args = parser.parse_args()

def read_file_or_blank(path):
    if not path:
        return ""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Warning: couldn't read {path}: {e}")
        return ""

incident_description = read_file_or_blank(args.desc)
parties_involved = read_file_or_blank(args.involved)
extra_info = read_file_or_blank(args.syllabus)

# === CONFIG ===
HARDCODED = {
    "url": "https://berkeley-advocate.symplicity.com/public_report/index.php/pid413831?",
    "reporter_type": "Faculty",
    "reporter_name": "Justin Yokota",
    "reporter_email": "cs61b-misconduct@berkeley.edu",
    "reporter_phone": "123-456-7890",
    "incident_type": "Academic Misconduct",
    "incident_date": "2025-05-11",
    "incident_hour": "03",
    "incident_minute": "30",
    "incident_ampm": "PM",
    "location": "example text",
}

# === SETUP BROWSER ===
driver = webdriver.Chrome()
driver.get(HARDCODED["url"])
wait = WebDriverWait(driver, 10)

def select_by_text(name, value):
    sel = Select(wait.until(EC.element_to_be_clickable((By.NAME, f"dnf_class_values[incident][{name}]"))))
    sel.select_by_visible_text(value)

def fill_input(name, value):
    el = wait.until(EC.presence_of_element_located((By.NAME, f"dnf_class_values[incident][{name}]")))
    el.clear()
    el.send_keys(value)

def fill_textarea(name, value):
    el = wait.until(EC.presence_of_element_located((By.NAME, f"dnf_class_values[incident][{name}]")))
    el.clear()
    el.send_keys(value)

# === FILL FORM ===
select_by_text("reporter_type", HARDCODED["reporter_type"])
fill_input("reporter_name", HARDCODED["reporter_name"])
fill_input("reporter_email", HARDCODED["reporter_email"])
fill_input("reporter_phone", HARDCODED["reporter_phone"])
select_by_text("public_report__incident_type", HARDCODED["incident_type"])
fill_textarea("extra_info", HARDCODED["location"])

# === Fill optional fields if provided ===
if incident_description:
    fill_textarea("description", incident_description)

if parties_involved:
    fill_textarea("other_student", parties_involved)

if extra_info:
    fill_textarea("extra_info", extra_info)

# === DATE + TIME ===
try:
    date_input = driver.find_element(By.ID, "dnf_class_values_incident__incident_date_")
    driver.execute_script("arguments[0].removeAttribute('readonly')", date_input)
    date_input.clear()
    date_input.send_keys(HARDCODED["incident_date"])

    Select(driver.find_element(By.NAME, "_hour_dnf_class_values[incident][incident_date]")).select_by_visible_text(HARDCODED["incident_hour"])
    Select(driver.find_element(By.NAME, "_min_dnf_class_values[incident][incident_date]")).select_by_visible_text(HARDCODED["incident_minute"])
    Select(driver.find_element(By.NAME, "_ampm_dnf_class_values[incident][incident_date]")).select_by_visible_text(HARDCODED["incident_ampm"])
except Exception as e:
    print("Error setting date/time:", e)

# === CONFIRM ===
input("🚨 Press ENTER to submit the form (or Ctrl+C to cancel)...")

# === SUBMIT ===
try:
    submit = driver.find_element(By.CSS_SELECTOR, "form input[type='submit']")
    submit.click()
    print("✅ Form submitted.")
except Exception as e:
    print("❌ Submit failed:", e)

time.sleep(5)
driver.quit()
