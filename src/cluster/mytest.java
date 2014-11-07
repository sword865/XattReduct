package cluster;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.ui.ApplicationFrame;

import weka.core.Instances;

public class mytest {

	
	
	public static void main(String[] args) throws Exception {
		File f = new File(
				"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\soybean.arff");
		// set data begin
		Instances dataset = new Instances(new FileReader(f));
		for(int i=0;i<dataset.numAttributes();i++){
			dataset.deleteWithMissing(i);
		}
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		int[] klable=new int[dataset.numInstances()];
		int K=dataset.numClasses();
		double[] rskiseq=new double[100];
		double[] rskmseq=new double[100];
		
		for(int i=0;i<100;i++){
			dataset.randomize(new Random(i));
			double rski=k_modesImprove.kModes(dataset, K, klable,new double[3],new Random(i));
			double rskm=k_modes.kModes(dataset, K, klable,new double[3],new Random(i));
			rskiseq[i]=rski;
			rskmseq[i]=rskm;
		}
		JFreeChart chart =Myploter.draw2valueseq(rskiseq,rskmseq,"Ki","Kmode","data","X","Y");
		ApplicationFrame frame = new ApplicationFrame("Title");
		ChartPanel chartPanel = new ChartPanel(chart);
		chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
		frame.setContentPane(chartPanel);
		frame.pack();
		frame.setVisible(true);
	}
}
