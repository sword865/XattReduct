package cluster;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.jfree.chart.JFreeChart;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import cluster.Ki.CentreswithFreq;

public class FuzzyKi extends Ki {

	double finalvalue = 0;
	double[][] weight = null;
	static double  para = 1.1;
	static XSSFWorkbook book;
	static XSSFSheet sheet;
	static XSSFRow row;

	public void computefuzzyvaluate(Instances datawithclass,double par) throws Exception{
		fXB=ClusterEvaluate.getfuzzyXB(datawithclass, m_ClusterCentroids, 
				weight, par);
		fFS=ClusterEvaluate.getfuzzyFS(datawithclass, m_ClusterCentroids, 
				weight, par);
		Instances newdata=new Instances(datawithclass);
		newdata.setClassIndex(-1);
		newdata.deleteAttributeAt(datawithclass.numAttributes()-1);
		fopt=ClusterEvaluate.getOptFun_f(newdata, m_ClusterCentroids, 
				weight, par);
	}
	
	public CentreswithFreq getNewCentres_KI(Instances data, double[][] weight,
			int K) {
		CentreswithFreq cwf = new CentreswithFreq();
		double[][] freq = new double[K][data.numAttributes()];
		for (int i = 0; i < freq.length; i++)
			for (int j = 0; j < freq[0].length; j++)
				freq[i][j] = 0;
		Instances centres = new Instances(data, K);
		for (int i = 0; i < K; i++) {
			Instance tmp = new Instance(data.numAttributes());
			for (int k = 0; k < data.numAttributes(); k++) {
				int numvalue=0;
				if(data.attribute(k).isNominal()){
					numvalue=data.attributeStats(k).distinctCount;
					numvalue=Math.max(data.attribute(k).numValues(), numvalue);
				}else{
					numvalue=(int)data.attributeStats(k).numericStats.max;
				}
				double[] count = new double[numvalue+1];//.attribute(k).numValues()
				Arrays.fill(count, 0);
				double sumcount = 0;
				for (int j = 0; j < data.numInstances(); j++) {
					int cur = (int) (1e-5 + data.instance(j).value(k));
					// System.out.printf(weight[j][i]+ " "+
					// Math.pow(weight[j][i],para)+"\n");
					count[cur] += Math.pow(weight[j][i], para);
					sumcount += Math.pow(weight[j][i], para);
				}
				int sel = Utils.maxIndex(count);
				tmp.setValue(k, sel);
				freq[i][k] = count[sel];
				freq[i][k] /= sumcount;
				if (Double.isNaN(freq[i][k])) {
					freq[i][k] = 1;
				}
			}
			centres.add(tmp);
		}
		cwf.centres = centres;
		cwf.freq = freq;
		return cwf;
	}

	public double optfun_ki_f(Instances data, CentreswithFreq cwf,
			double[][] weight) {
		double fvalue = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < cwf.centres.numInstances(); j++) {
				double dis = 0;
				for (int k = 0; k < data.numAttributes(); k++) {
					if (data.instance(i).value(k) != cwf.centres.instance(j)
							.value(k)) {
						dis += 1;
					} else {
						dis += 1 - cwf.freq[j][k];
					}
				}
				fvalue += Math.pow(weight[i][j], para) * dis;
			}

		}
		return fvalue;
	}

	public double[][] getWeight(CentreswithFreq cwf, Instances data) {
		double[][] ourweight = new double[data.numInstances()][cwf.centres
				.numInstances()];
		double[] dis = new double[cwf.centres.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			int flag = -1;
			for (int j = 0; j < cwf.centres.numInstances(); j++) {
				dis[j] = 0;
				for (int k = 0; k < data.numAttributes(); k++) {
					if (data.instance(i).value(k) != cwf.centres.instance(j)
							.value(k)) {
						dis[j] += 1;
					} else {
						dis[j] += 1 - cwf.freq[j][k];
					}
				}
				if (dis[j] == 0) {
					flag = j;
					break;
				}
			}
			if (flag < 0) {
				if (para > 1) {
					for (int j = 0; j < cwf.centres.numInstances(); j++) {
						double sum = 0;
						for (int k = 0; k < cwf.centres.numInstances(); k++) {
							sum += Math
									.pow((dis[j] / dis[k]), 1.0 / (para - 1));
						}
						ourweight[i][j] = 1 / sum;
					}
				} else {
					int sel = Utils.minIndex(dis);
					for (int j = 0; j < cwf.centres.numInstances(); j++) {
						if (j != sel) {
							ourweight[i][j] = 0;
						} else {
							ourweight[i][j] = 1;
						}
					}
				}
			} else {
				for (int j = 0; j < cwf.centres.numInstances(); j++) {
					if (j != flag) {
						ourweight[i][j] = 0;
					} else {
						ourweight[i][j] = 1;
					}
				}
			}
		}
		return ourweight;
	}

	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		int itercount = 0;
		optvalueseq = new Vector<Double>();
		optvalueseq_iter = new Vector<Double>();
		CentreswithFreq cwf = new CentreswithFreq();
		cwf.centres = randomCentres(data, m_NumClusters, random);
		cwf.freq = new double[cwf.centres.numInstances()][data.numAttributes()];
		for (int i = 0; i < cwf.freq.length; i++) {
			for (int j = 0; j < cwf.freq[0].length; j++) {
				cwf.freq[i][j] = 1;
			}
		}
		weight = getWeight(cwf, data);
		double prefunvalue = -1;
		double optfunvalue = prefunvalue;
		for (int iter = 0; iter < maxiter; iter++) {
			itercount++;
			cwf = getNewCentres_KI(data, weight, m_NumClusters);
			weight = getWeight(cwf, data);
			optfunvalue = optfun_ki_f(data, cwf, weight);
			// optvalueseq.add(optfun(data,cwf.centres,m_clusters));
			// optvalueseq_iter.add(optfunvalue);
			if (prefunvalue > -1 && optfunvalue - prefunvalue > -1e-4)
				break;
			prefunvalue = optfunvalue;
		}
		m_clusters = new int[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			m_clusters[i] = Utils.maxIndex(weight[i]);
		}
		finalvalue = optfun(data, cwf.centres, m_clusters);
		// System.out.print(itercount);
		m_ClusterCentroids = new Instances(cwf.centres);
	}

	public double getbestvalue() {
		return finalvalue;
	}

	public static void partest() throws Exception {
		//,"arrhythmia","breast-cancer","cleveland","cmc","ecoli","balance-scale"
		String[] datastr = {"arcene"};
		//"bridges_version1_r","bridges_version2_r","heart-h","cylinder-bands","anneal","heart-c","postoperative-patient-data"
		//"data10_10_5_8_5_1000","data20_5_5_8_5_1000","data30_10_5_10_5_1000","data50_10_5_10_5_1000"};
		//"Monk_1","Monk_2","Monk_3",
		//"balance-scale","molecular-biology_promoters_lastclass_withid","breast-cancer","kr-vs-kp","mushroom" ,"nursery"};//"zoo","soybean_small","data20_10_5_10_10_500"
//, , "tic-tac-toe","lymph","soybean",
//		"mushroom" ,"kr-vs-kp", "data50_5_5_10_5_300", "data15_3_3_10_10_500","data_100_20_5_10_10_500",
		//		String[] datastr = {};
// 
		for (int dataset_iter = 0; dataset_iter < datastr.length; dataset_iter++) {
			String dataname = datastr[dataset_iter];
			File f = new File(
					"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\"
							+ dataname + ".arff");
			// set data begin
			Instances dataset = new Instances(new FileReader(f));
			for (int i = 0; i < dataset.numAttributes(); i++) {
				dataset.deleteWithMissing(i);
			}
			dataset.setClassIndex(dataset.numAttributes()-1);
			Discretize discretize = new Discretize();   
			discretize.setInputFormat(dataset);
			dataset = Filter.useFilter( dataset, discretize);
			
			Instances newdataset = new Instances(dataset);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			newdataset.setClassIndex(-1);
			newdataset.deleteAttributeAt(newdataset.numAttributes() - 1);
			int K = dataset.attributeStats(dataset.numAttributes()-1).distinctCount;//.classAttribute()..numValues();
			final int testnum=10;
			int T = testnum;
//			int wincount = 0;
			double[] parseq=new double[11];
			for(int i=0;i<parseq.length;i++)
				parseq[i]=1+0.02*i;

			double[][] fXBseq = new double[2][parseq.length];
			double[][] fFSseq = new double[2][parseq.length];
			double[][] foptseq = new double[2][parseq.length];
			double[][] XBseq=new double[2][parseq.length];
			double[][] Ikseq=new double[2][parseq.length];
			double[][] NMIseq=new double[2][parseq.length];
			int[][] ACnumseq=new int[2][parseq.length];
			double[][] accuracyseq=new double[2][parseq.length];
			for(int k=0;k<2;k++){
				for(int j=0;j<parseq.length;j++){
					fXBseq[k][j] = 0;//new double[2][parseq.length];
					fFSseq[k][j] = 0;//new double[2][parseq.length];
					foptseq[k][j] = 0;//new double[2][parseq.length];
					XBseq[k][j]=0;//new double[2][parseq.length];
					Ikseq[k][j]=0;//new double[2][parseq.length];
					NMIseq[k][j]=0;//new double[2][parseq.length];
					ACnumseq[k][j]=0;//new int[2][parseq.length];
					accuracyseq[k][j]=0;
				}
			}						
			int pos = -1;
			for(int i=0;i<parseq.length;i++){
				pos=i;
				T = testnum;
				while ((T--) > 0) {
					int r = T;//new Random().nextInt();// T;
					FuzzyKi fkicluster = new FuzzyKi();
					fkicluster.para=parseq[i];
					fkicluster.setNumClusters(K);
					fkicluster.setRandom(new Random(r));
					fkicluster.getDistanceFun().setInstances(newdataset);
					fkicluster.buildClusterer(newdataset);
					fkicluster.computevaluate(dataset);
					fkicluster.computefuzzyvaluate(dataset,para);
					double fkivalue = fkicluster.getbestvalue();
					XBseq[0][pos] += fkicluster.getXB();
					Ikseq[0][pos] += fkicluster.getIk();
					NMIseq[0][pos] += fkicluster.getNMI();
					fXBseq[0][pos] += fkicluster.getfXB();
					fFSseq[0][pos] += fkicluster.getfFS();
					foptseq[0][pos] += fkicluster.getfopt();
					ACnumseq[0][pos]+= fkicluster.getACnumber();
					// System.out.printf("Fuzzy KI:"+ fkivalue +"\n");
				
				
					FuzzyKmode fkmodecluster = new FuzzyKmode();
					fkmodecluster.para=parseq[i];
					fkmodecluster.setNumClusters(K);
					fkmodecluster.setRandom(new Random(r));
					fkmodecluster.getDistanceFun().setInstances(newdataset);
					fkmodecluster.buildClusterer(newdataset);
					fkmodecluster.computevaluate(dataset);
					fkmodecluster.computefuzzyvaluate(dataset,para);
					double fkmodevalue = fkmodecluster.getbestvalue();
					XBseq[1][pos] += fkmodecluster.getXB();
					Ikseq[1][pos] += fkmodecluster.getIk();
					NMIseq[1][pos] += fkmodecluster.getNMI();
					fXBseq[1][pos] += fkmodecluster.getfXB();
					fFSseq[1][pos] += fkmodecluster.getfFS();
					foptseq[1][pos] += fkmodecluster.getfopt();
					ACnumseq[1][pos] += fkmodecluster.getACnumber();
				}
			}
			for(int k=0;k<2;k++){
				for(int j=0;j<parseq.length;j++){
					accuracyseq[k][j]=((double)ACnumseq[k][j]/(double)dataset.numInstances());
				}
			}
			for(int k=0;k<2;k++){
				for(int j=0;j<parseq.length;j++){
					fXBseq[k][j] /= testnum;//new double[2][parseq.length];
					fFSseq[k][j] /= testnum;//new double[2][parseq.length];
					foptseq[k][j] /= testnum;//new double[2][parseq.length];
					XBseq[k][j]/=testnum;//new double[2][parseq.length];
					Ikseq[k][j]/=testnum;//new double[2][parseq.length];
					NMIseq[k][j]/=testnum;//new double[2][parseq.length];
					ACnumseq[k][j]/=testnum;//new int[2][parseq.length];
					accuracyseq[k][j]/=testnum;
				}
			}			

			
			String[] namestr = { "FuzzyKI","FuzzyKmode"};
			JFreeChart chart = Myploter.drawvalueseq(XBseq, namestr, dataname,
					"", "XB",parseq);
	//		Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_XB.jpg", 800,
	//				600);
			chart = Myploter.drawvalueseq(Ikseq, namestr, dataname, "", "Ik",parseq);
	//		Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_Ik.jpg", 800,
	//				600);
			chart = Myploter.drawvalueseq(NMIseq, namestr, dataname, "", "NMI",parseq);
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_NMI.jpg", 800,
					600);
			chart = Myploter.drawvalueseq(fXBseq, namestr, dataname, "", "fXB",parseq);
	//		Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_fXB.jpg", 800,
	//				600);
			chart = Myploter.drawvalueseq(fFSseq, namestr, dataname, "", "fFS",parseq);
	//		Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_fFS.jpg", 800,
	//				600);
			chart = Myploter.drawvalueseq(foptseq, namestr, dataname, "", "Optfun",parseq);
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_Optfun.jpg", 800,
					600);
			chart = Myploter.drawvalueseq(accuracyseq, namestr, dataname, "", "AC",parseq);
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_par_AC.jpg", 800,
					600);
			for(int i=0;i<namestr.length;i++){
				chart = Myploter.drawfreqhist(ACnumseq[i],0,dataset.numInstances(),dataname,100);
	//			Myploter.saveAsFile(chart, "ans\\" + dataname + "_"+ namestr[i] + "_AChist.jpg", 800,
	//				600);
			}
			
			book = new XSSFWorkbook();
			int curline = 0;
			row = book.createSheet("XB").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("Ik").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("NMI").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fXB").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fFS").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fOpt").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("ACnumber").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			
			for(int i=0;i<fXBseq[0].length;i++){
				curline++;
				row=book.getSheetAt(0).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(XBseq[j][i]);
				}
				row=book.getSheetAt(1).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(Ikseq[j][i]);
				}
				row=book.getSheetAt(2).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(NMIseq[j][i]);
				}
				row=book.getSheetAt(3).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(fXBseq[j][i]);
				}
				row=book.getSheetAt(4).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(fFSseq[j][i]);
				}
				row=book.getSheetAt(5).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(foptseq[j][i]);
				}
				row=book.getSheetAt(6).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(ACnumseq[j][i]);
				}
			}
			FileOutputStream fileOut;
			fileOut = new FileOutputStream("ans\\FuzzyKI_par_"+ dataname  + ".xlsx");
			book.write(fileOut);
			fileOut.close();
		}
		System.out.println("写入成功，运行结束！");		
	}
	
	public static void comparetest() throws Exception {
//,
//		String[] datastr = {"zoo", "vote", "soybean", "car","tic-tac-toe","lymph","soybean",
//				"data50_5_5_10_5_300", "data15_3_3_10_10_500", "mushroom" ,"kr-vs-kp"};
		//"arrhythmia","breast-cancer","cleveland","cmc","ecoli","balance-scale"
		String[] datastr = {"arcene"};
		//"bridges_version1_r","bridges_version2_r","heart-h","cylinder-bands","anneal","heart-c","postoperative-patient-data"};
		// "data50_5_5_10_5_300","data15_3_3_10_10_500", "zoo", "vote","tic-tac-toe","lymph","primary-tumor","hepatitis","dermatology", "soybean","car"
		for (int dataset_iter = 0; dataset_iter < datastr.length; dataset_iter++) {
			String dataname = datastr[dataset_iter];
			File f = new File(
					"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\"
							+ dataname + ".arff");
			// set data begin
			Instances dataset = new Instances(new FileReader(f));
			for (int i = 0; i < dataset.numAttributes(); i++) {
				dataset.deleteWithMissing(i);
			}
			dataset.setClassIndex(dataset.numAttributes()-1);
			Discretize discretize = new Discretize();   
			discretize.setInputFormat(dataset);
			dataset = Filter.useFilter( dataset, discretize);
			
			Instances newdataset = new Instances(dataset);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			newdataset.setClassIndex(-1);
			newdataset.deleteAttributeAt(newdataset.numAttributes() - 1);
			int K = dataset.attributeStats(dataset.numAttributes()-1).distinctCount;//.classAttribute().numValues();
			final int testnum=100;
			int T = testnum;
//			int wincount = 0;
			double[][] fXBseq = new double[4][T];
			double[][] fFSseq = new double[4][T];
			double[][] foptseq = new double[4][T];
			double[][] XBseq=new double[4][T];
			double[][] Ikseq=new double[4][T];
			double[][] NMIseq=new double[4][T];
			int[][] ACnumseq=new int[4][T];
			double[][] accuracyseq=new double[4][T];
			int pos = -1;
			while ((T--) > 0) {
				pos++;
				int r = new Random().nextInt();// T;
				FuzzyKi fkicluster = new FuzzyKi();
				fkicluster.setNumClusters(K);
				fkicluster.setRandom(new Random(r));
				fkicluster.getDistanceFun().setInstances(newdataset);
				fkicluster.buildClusterer(newdataset);
				fkicluster.computevaluate(dataset);
				fkicluster.computefuzzyvaluate(dataset,para);
				double fkivalue = fkicluster.getbestvalue();
				XBseq[0][pos]=fkicluster.getXB();
				Ikseq[0][pos]=fkicluster.getIk();
				NMIseq[0][pos]=fkicluster.getNMI();
				fXBseq[0][pos] = fkicluster.getfXB();
				fFSseq[0][pos] = fkicluster.getfFS();
				foptseq[0][pos] = fkicluster.getfopt();
				ACnumseq[0][pos]=fkicluster.getACnumber();
				// System.out.printf("Fuzzy KI:"+ fkivalue +"\n");
				
				Ki kicluster = new Ki();
				kicluster.setNumClusters(K);
				kicluster.setRandom(new Random(r));
				kicluster.getDistanceFun().setInstances(newdataset);
				kicluster.buildClusterer(newdataset);
				kicluster.computevaluate(dataset);
				kicluster.computefuzzyvaluate(dataset,para);
				double kivalue = kicluster.getbestvalue();
				XBseq[1][pos]=kicluster.getXB();
				Ikseq[1][pos]=kicluster.getIk();
				NMIseq[1][pos]=kicluster.getNMI();
				fXBseq[1][pos] = kicluster.getfXB();
				fFSseq[1][pos] = kicluster.getfFS();
				foptseq[1][pos] = kicluster.getfopt();
				ACnumseq[1][pos]=kicluster.getACnumber();
				
				FuzzyKmode fkmodecluster = new FuzzyKmode();
				fkmodecluster.setNumClusters(K);
				fkmodecluster.setRandom(new Random(r));
				fkmodecluster.getDistanceFun().setInstances(newdataset);
				fkmodecluster.buildClusterer(newdataset);
				fkmodecluster.computevaluate(dataset);
				fkmodecluster.computefuzzyvaluate(dataset,para);
				double fkmodevalue = fkmodecluster.getbestvalue();
				XBseq[2][pos]=fkmodecluster.getXB();
				Ikseq[2][pos]=fkmodecluster.getIk();
				NMIseq[2][pos]=fkmodecluster.getNMI();
				fXBseq[2][pos] = fkmodecluster.getfXB();
				fFSseq[2][pos] = fkmodecluster.getfFS();
				foptseq[2][pos] = fkmodecluster.getfopt();
				ACnumseq[2][pos]=fkmodecluster.getACnumber();
				
				Kmode kmodecluster = new Kmode();
				kmodecluster.setNumClusters(K);
				kmodecluster.setRandom(new Random(r));
				kmodecluster.getDistanceFun().setInstances(newdataset);
				kmodecluster.buildClusterer(newdataset);
				kmodecluster.computevaluate(dataset);
				kmodecluster.computefuzzyvaluate(dataset,para);
				double kmodevalue = kmodecluster.getbestvalue();
				XBseq[3][pos]=kmodecluster.getXB();
				Ikseq[3][pos]=kmodecluster.getIk();
				NMIseq[3][pos]=kmodecluster.getNMI();
				fXBseq[3][pos] = kmodecluster.getfXB();
				fFSseq[3][pos] = kmodecluster.getfFS();
				foptseq[3][pos] = kmodecluster.getfopt();
				ACnumseq[3][pos] = kmodecluster.getACnumber();
			}
			for(int i=0;i<accuracyseq.length;i++){
				for(int j=0;j<accuracyseq[0].length;j++){
					accuracyseq[i][j]=((double)ACnumseq[i][j]/(double)dataset.numInstances());
				}
			}
			String[] namestr = { "FuzzyKI","KI","FuzzyKmode","Kmode"  };
			JFreeChart chart = Myploter.drawvalueseq(XBseq, namestr, dataname,
					"", "XB");
		//	Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_XB.jpg", 800,
		//			600);
			chart = Myploter.drawvalueseq(Ikseq, namestr, dataname, "", "Ik");
		//	Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_Ik.jpg", 800,
		//			600);
			chart = Myploter.drawvalueseq(NMIseq, namestr, dataname, "", "NMI");
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_NMI.jpg", 800,
					600);
			chart = Myploter.drawvalueseq(fXBseq, namestr, dataname, "", "fXB");
		//	Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_fXB.jpg", 800,
		//			600);
			chart = Myploter.drawvalueseq(fFSseq, namestr, dataname, "", "fFS");
		//	Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_fFS.jpg", 800,
		//			600);
			chart = Myploter.drawvalueseq(foptseq, namestr, dataname, "", "Optfun");
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_Optfun.jpg", 800,
					600);
			chart = Myploter.drawvalueseq(accuracyseq, namestr, dataname, "", "AC");
			Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_AC.jpg", 800,
					600);
			for(int i=0;i<namestr.length;i++){
				chart = Myploter.drawfreqhist(ACnumseq[i],0,dataset.numInstances(),namestr[i],100);
		//		Myploter.saveAsFile(chart, "ans\\" + dataname + "_"+ namestr[i] + "_AChist.jpg", 800,
		//			600);
			}
			
			book = new XSSFWorkbook();
			int curline = 0;
			row = book.createSheet("XB").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("Ik").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("NMI").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fXB").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fFS").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("fOpt").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			row = book.createSheet("ACnumber").createRow(curline);
			for(int i=0;i<namestr.length;i++){
				row.createCell(i).setCellValue(namestr[i]);
			}
			
			for(int i=0;i<fXBseq[0].length;i++){
				curline++;
				row=book.getSheetAt(0).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(XBseq[j][i]);
				}
				row=book.getSheetAt(1).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(Ikseq[j][i]);
				}
				row=book.getSheetAt(2).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(NMIseq[j][i]);
				}
				row=book.getSheetAt(3).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(fXBseq[j][i]);
				}
				row=book.getSheetAt(4).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(fFSseq[j][i]);
				}
				row=book.getSheetAt(5).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(foptseq[j][i]);
				}
				row=book.getSheetAt(6).createRow(curline);
				for(int j=0;j<fXBseq.length;j++){
					row.createCell(j).setCellValue(ACnumseq[j][i]);
				}
			}

			FileOutputStream fileOut;
			fileOut = new FileOutputStream("ans\\FuzzyKI_"+ dataname  + ".xlsx");
			book.write(fileOut);
			fileOut.close();
		}
		System.out.println("写入成功，运行结束！");
	}
	
	public static void main(String[] args) throws Exception {
		partest();
		comparetest();
	}
}
