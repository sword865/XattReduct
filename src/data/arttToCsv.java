package data;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;

public class arttToCsv {
	public static void main(String[] args) throws IOException {
		String dirname = "D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\data\\";
		File rsfile = new File(dirname + "convertRs\\");
		if (!rsfile.exists()) {
			rsfile.mkdir();
		}
		File files = new File(dirname);
		File[] contents = files.listFiles();
		for (File filename : contents) {
			if (filename.isDirectory())
				continue;
			FileReader fr = new FileReader(filename.getAbsolutePath());
			Instances data = new Instances(fr);
			// String
			// dataname="D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\breast-cancer.arff";
			// Instances data = new Instances(new FileReader(dataname));
			run(data, rsfile);
		}
	}

	static void run(Instances data, File rsfile) throws IOException {
		for(int i=0;i<data.numInstances();i++){
			for(int j=0;j<data.numAttributes();j++){
				if(data.instance(i).isMissing(j)){
					data.instance(i).setValue(j, data.meanOrMode(j));
				}
			}
		}
		
		FileWriter fileWriter = new FileWriter(rsfile.getAbsoluteFile() + "\\"
				+ data.relationName());
		fileWriter.append("data set name:" + data.relationName() + "\r\n");
		// fileWriter.append("data type:"+data.+"\r\n");
		fileWriter.append(data.relationName() + "\r\n");
		int j;
		fileWriter.append("num value:");
		fileWriter.append(data.relationName() + "\r\n");
		for (j = 0; j < data.numAttributes(); ++j) {
			fileWriter.append("" + (int) data.attribute(j).numValues() + ",");
		}
		fileWriter.append(data.relationName() + "\r\n");
		for (int i = 0; i < data.numInstances(); ++i) {
			for (j = 0; j < data.numAttributes() - 1; ++j) {
				if (data.instance(i).isMissing(j))
					fileWriter.append("" + -1 + ",");
				else
					fileWriter.append("" + (int) data.instance(i).value(j)
							+ ",");
			}
			fileWriter.append("" + (int) (data.instance(i).value(j) + 1)
					+ "\r\n");
		}

		fileWriter.flush();
		fileWriter.close();

	}

}
