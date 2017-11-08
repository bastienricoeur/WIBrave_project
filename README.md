## WI Brave

The project is provided **pre-built** as an **executable jar** but can be built using `sbt assembly` at the project's root.

To run the jar in **scala** you need to use **scala 2.11**
### Run the project
Clone this repository (or download as zip)
- `git clone https://github.com/bastienricoeur/WIBrave_project`
- `cd WIBrave_project/dist`

*The `dist/` folder contains the jar executable and the `model/` folder*

***To display usage help***
- `java -jar wibrave_project.jar --help`

*The model is pre-trained with 1.000.000 data so it can be used directly.*

***To predict a batch against the model, run:***
- `java -jar wibrave_project.jar predict --input /path/to/input.json --model /path/to/modelDir --output /path/to/output`

*This will output a csv file with predicted labels in the specified output directory.*

***To test the model with data [Label column needed]***
- `java -jar wibrave_project.jar test --input /path/to/input.json --model /path/to/modelDir --output /path/to/output`

*This will output a csv file with predicted labels as well as original labels in the specified output directory*

*This command will also log statistics about the test based on the original labels and predictions*

***To train a new model with data [Label column needed]***
- `java -jar wibrave_project.jar train --input /path/to/input.json --output /path/to/model`

*This will output a model directory that can be loaded with `test` or `predict`.*

### Build the project

To build the project, you can use
- `sbt assembly`
at the root of the project, this will generate a new jar file


***Built with :heart: by Brave Team***
