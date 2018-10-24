import gspread
from oauth2client.service_account import ServiceAccountCredentials
get_credentials = ServiceAccountCredentials.from_json_keyfile_name

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
# To make this work, obtain credentials from Google Sheets API and save to
# creds.json in current directory.
credentials = get_credentials('creds.json', scope)
gc = gspread.authorize(credentials)
sheet_name = 'pytorch-generative'


def upload_to_google_sheets(row_data, index=2):
    worksheet = gc.open(sheet_name).sheet1
    worksheet.insert_row(row_data, index=index)
