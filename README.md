# thunderbird-addon-spam-detector

## Run
1. Make sure you have python3 installed
2. Run the script "build.sh"
3. Add the Add-on to thunderbird
    + Settings
    + Add-ons and Themes
    + Click the cogwheel
    + Select Debug Add-ons
    + Load Tempoary Add-on
    + Choose the Manifest of the Repository
4. Add the Full Path of your script located at src/script.py(e.g "path/src/script.py") and Download the Manifest as instructed
5. Choose Extended and Conection Based
6. Choose the folders you want to be tracked

## Resources:
+ [thunderbird-addon-scriptable-notifications](https://github.com/electrotype/thunderbird-addon-scriptable-notifications)
+ [messages API](https://webextension-api.thunderbird.net/en/stable/messages.html#messages-messageproperties)