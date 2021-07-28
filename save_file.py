import os
import pandas as pd
from openpyxl import load_workbook
import xlsxwriter
from datetime import datetime, date, time
import xlrd


class save_data():
    def __init__(self, now, Code, State, Score, Speed, model):
        self.now = now
        self.Code = Code
        self.State = State
        self.Score = Score
        self.Speed = Speed
        self.model = model
        self.path = './output/Info/' + self.model + "/" + self.model + "_" + self.now.strftime("%m") + \
                    self.now.strftime("%Y") + '.xlsx'
        self.path2 = r'./output/Info/' + self.model + "/" + self.model + "_" + self.now.strftime("%m") + \
                     self.now.strftime("%Y") + '.xlsx'
        self.message = self.save_xlsx()

    def save_xlsx(self):
        while True:
            try:
                if not os.path.isfile(self.path):
                    # dataframe Name and Age columns
                    df = pd.DataFrame(columns=["Date", "Time", "Code", "State", "Score", "Speed"])
                    # Create a Pandas Excel writer using XlsxWriter as the engine.
                    writer = pd.ExcelWriter(self.path, engine='xlsxwriter')

                    # Convert the dataframe to an XlsxWriter Excel object.
                    df.to_excel(writer, index=False)
                    worksheet = writer.sheets['Sheet1']
                    worksheet.set_column('A:A', 20)
                    worksheet.set_column('B:B', 10)
                    worksheet.set_column('C:C', 30)

                    # Close the Pandas Excel writer and output the Excel file.
                    writer.save()

                reader = pd.read_excel(self.path2)
                df = pd.DataFrame([[self.now.strftime("%d %B %Y"), self.now.strftime("%X"),
                                   self.Code, self.State, self.Score, self.Speed]],
                                  columns=["Date", "Time", "Code", "State", "Score", "Speed"])

                with pd.ExcelWriter(self.path, engine="openpyxl", mode="a", date_format='dd mmmm yyyy') as writer:
                    # try to open an existing workbook
                    writer.book = load_workbook(self.path)
                    # copy existing sheets
                    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
                    print(df)
                    df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
                    # Close the Pandas Excel writer and output the Excel file.
                    writer.save()
                    return None
                break
            except Exception as e:
                s = str(e)
                print(s)
                message = "The action can't be completed because the file is open in another tab!\nClose file and try " \
                          "again... "
                return message
                raise
                break


if __name__ == "__main__":
    now = datetime.now()
    today = datetime.today()
    save_data(now, "abc", "abc", 1.2, 1.2, "abc")
