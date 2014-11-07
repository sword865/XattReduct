package macLn.src;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;

import weka.core.Instances;
import weka.core.converters.ArffSaver;


public class creatFoldsData {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
			getDatas();
	}
	
	public static void getDatas() throws Exception
	{
		Instances data=k_modesExp2.loadFile("E:\\dataset\\cate\\objects\\cc.arff");
		Vector<Instances> v=getFoldsData(data,5,12);
		saveToFiles(v.get(0),"D:\\rs\\f0.arff");
		for(int i=1;i<5;++i){
			for(int j=0;j<v.get(i).numInstances();++j){
				v.get(0).add(v.get(i).instance(j));
			}
			String f="D:\\rs\\f"+i+".arff";
			saveToFiles(v.get(0),f);
		}
		
	}

	public static Vector<Instances> getFoldsData(Instances data,int folds,int seed)
	{
		Vector<Instances> foldRes= new Vector<Instances>(folds);
		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand);         // randomize data with number generator
		for (int n = 0; n < folds; n++) 
		{
			  // Instances train = randData.trainCV(folds, n);
			   Instances test = randData.testCV(folds, n);
			   foldRes.add(test);
		}
		return foldRes;
	}
	
	public static void saveToFiles(Instances dataSet,String fileDes) throws IOException
	{
		ArffSaver saver = new ArffSaver();
		 saver.setInstances(dataSet);
		 saver.setFile(new File(fileDes));
		 //saver.setDestination(new File(fileDes));   // **not** necessary in 3.5.4 and later
		 saver.writeBatch();
	}
}
