package data;

import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;
import java.util.Vector;

//分类的几种
import weka.datagenerators.classifiers.classification.Agrawal;
import weka.datagenerators.classifiers.classification.BayesNet;
import weka.datagenerators.classifiers.classification.LED24;
import weka.datagenerators.classifiers.classification.RandomRBF;
import weka.datagenerators.classifiers.classification.RDG1;
//回归
import weka.datagenerators.classifiers.regression.Expression;
import weka.datagenerators.classifiers.regression.MexicanHat;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
public class inputData {

	/**
	 * @param args
	 * 从控制台读入命令行参数，返回一个Vector<Instances>
	 * -F file   从file文件输入数据
	 * -G mode   以mode方式生成数据
	 * -C n      将数据等分成n份
	 * 例如-F D:\dataset\zoo.arff -C 10
	 * 
	 */
	public static  void main(String[] args)throws Exception {
		// TODO Auto-generated method stub
		getData();
	}
	
	public static Vector<Instances> getData()throws Exception
	{
		String tmpStr;
		Instances data=null;
		String[] paras=getOptions();
		tmpStr = Utils.getOption('F', paras);
		if(tmpStr.length()>0){
			data=getRawData('F',tmpStr);
		}else{
			tmpStr = Utils.getOption('G', paras);
			if(tmpStr.length()>0){
				data=getRawData('G',tmpStr);
			}
		}
	
		Vector<Instances> dataVec=null;
		tmpStr=Utils.getOption('C', paras);             //C  cross  分成n等份
		if(tmpStr.length()>0){
			System.out.println(tmpStr);
			int folds=Integer.parseInt(tmpStr);
			processData(data,'C',folds);
		}else if(true){
			tmpStr=Utils.getOption('P', paras);
			if(tmpStr.length()>0){
				double percent=Double.parseDouble(tmpStr);
				processData(data,'P',percent);
			}
			
		}else{
			
		}
		
		tmpStr = Utils.getOption('G', paras);          // G计算机生成
		//test
		printData(dataVec);
		return dataVec;
	}
	
	public static Vector<Instances> processData(Instances data,char op,double ratio)
	{
		Vector<Instances> dataVec=null;
		if(op=='C'){
			Random rnd=new Random();
			int seed=rnd.nextInt();
			dataVec=getFoldsData(data,(int)ratio,seed);
		}else if(op=='P'){
			
		}else{
			dataVec= new Vector<Instances>(1);
			dataVec.add(data);
		}
		return dataVec;
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
	
	public static Instances getRawData(char op,String str)throws Exception
	{
		Instances data=null;                           //存储原始数据集
		if(op=='F'){
			System.out.println(str);
			DataSource source=new DataSource(str);        //可以输入artt  csv等文件格式
			data=source.getDataSet();
		}else if(op=='G'){
			System.out.println("input"+str+"’s parameter:");       
			String[] options=getOptions();            //获得数据生成函数的参数，再传递给Weka
			if(str.equals("agrawal")){
				data=agrawal(options);
			}else if(str.equals("bayesNet")){
				data=bayesNet(options);
			}else if(str.equals("led24")){
				data=led24(options);
			}
		}

		data.setClassIndex(data.numAttributes()-1);      //设置类标
		return data;
	}
	//获取参数，返回参数
	public static String[] getOptions() throws Exception
	{
		Scanner scanner=new Scanner(System.in);
		String options=scanner.nextLine();
		return Utils.splitOptions(options);
	}

	public static Instances agrawal(String[] options) throws Exception
	{
		Agrawal genertor=new Agrawal();
		genertor.setOptions(options);
		//需要setseed
		return genertor.generateExamples();
	}
	public static Instances bayesNet(String[] options) throws Exception
	{
		BayesNet genertor=new BayesNet();
		genertor.setOptions(options);
		//需要setseed
		return genertor.generateExamples();
	}
	
	public static Instances led24(String[] options) throws Exception
	{
		LED24 genertor=new LED24();
		genertor.setOptions(options);
		//需要setseed
		return genertor.generateExamples();
	}
	
	public static void printData(Vector<Instances> dataVec)
	{
		for(int i=0;i<dataVec.capacity();++i)
		{
			Instances tmp=dataVec.elementAt(i);
			System.out.println(tmp.numInstances());
			for(int j=0;j<tmp.numInstances();++j){
				System.out.print(tmp.instance(j));
			}
			System.out.println();
		}
	}
}
