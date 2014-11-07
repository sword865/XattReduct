package FS;

import java.awt.Color;
import java.awt.Font;
import java.io.FileInputStream;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Vector;


import org.apache.poi.ss.usermodel.WorkbookFactory;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.axis.TickUnitSource;
import org.jfree.chart.axis.TickUnits;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.HorizontalAlignment;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.VerticalAlignment;

import cluster.Myploter;

public class excelTopic {

	static final String[] methodname={"SemiEntropy","Tradition Entropy"};
	static final int methodnum=methodname.length;//6
		//{"Semi-rough-D ","Semi-rough-P ","Dismat based ","Unlabel rough","Entropy based","Pos based    "};
	static XSSFWorkbook book;
	static XSSFSheet sheet;
	static XSSFRow row;
	
	public static JFreeChart drawvaluevec(Vector<Vector<Double>> valueseq,String[] namestr,String dataname,String xlab,String ylab,double[] x)
			throws Exception {
		if (valueseq.size() != namestr.length)
			return null;
		XYSeries[] plotdata = new XYSeries[namestr.length];
		XYSeriesCollection plotdataset = new XYSeriesCollection();
		for (int i = 0; i < plotdata.length; i++) {
			plotdata[i] = new XYSeries(namestr[i]);
			for (int j = 0; j < valueseq.elementAt(i).size(); j++) {
				plotdata[i].add(x[j], valueseq.elementAt(i).elementAt(j).doubleValue());
			}
			plotdataset.addSeries(plotdata[i]);
		}
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart															// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		
		
		renderer.setSeriesPaint(2, Color.DARK_GRAY);
		renderer.setSeriesPaint(3, Color.GREEN);
		renderer.setSeriesPaint(4, Color.ORANGE);
		renderer.setSeriesPaint(5, Color.MAGENTA);
		
		renderer.setSeriesPaint(1, Color.red);
		renderer.setSeriesPaint(0, Color.BLUE);
		
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,18));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,18));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,18));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,18));
		plot.setBackgroundPaint(Color.WHITE);

		TickUnits units = new TickUnits();
		DecimalFormat df0 = new DecimalFormat("0");
		DecimalFormat df1 = new DecimalFormat("0");
		units.add(new NumberTickUnit(1.0D, df0));
		units.add(new NumberTickUnit(2D, df0));
		units.add(new NumberTickUnit(5D, df0));
		units.add(new NumberTickUnit(10D, df0));
		units.add(new NumberTickUnit(20D, df0));
		units.add(new NumberTickUnit(50D, df0));
		units.add(new NumberTickUnit(100D, df0));
		units.add(new NumberTickUnit(200D, df0));
		units.add(new NumberTickUnit(500D, df0));
		units.add(new NumberTickUnit(1000D, df1));
		
		NumberFormat numformatter = NumberFormat.getInstance();
		numformatter.setMaximumFractionDigits(2);
		numformatter.setMinimumFractionDigits(2);
		TickUnitSource source= (TickUnitSource)units;
		//((NumberAxis)plot.getRangeAxis()).setStandardTickUnits(source);//.setStandardTickUnits();	
		
		
		//plot.get
		chart.setBackgroundPaint(Color.WHITE);
		chart.getTitle().setFont(new Font("黑体",0,25));
		chart.getLegend(0).setItemFont(new Font("黑体",0,22));
		chart.getLegend(0).setMargin(new RectangleInsets(0,120,20,120));
		chart.getLegend(0).setItemLabelPadding(new RectangleInsets(1,10,1,10));
		chart.getLegend(0).setPadding(new RectangleInsets(5,5,5,5));
		return chart;
	}
	
	public static Vector<Vector<Double>> readdata(XSSFSheet sht){
		Vector<Vector<Double>> valueseq=new Vector<Vector<Double>>();
		for(int i=0;i<methodnum;i++){
			Vector<Double> v=new Vector<Double>();
			valueseq.add(v);
		}
		int[] labvec={2,5};
		for(int i=1;i<26;i++){
			row=sht.getRow(i);
			for(int j=0;j<methodnum;j++){
				XSSFCell cell=row.getCell(labvec[j]);
				if(cell!=null){
					valueseq.elementAt(j).add(cell.getNumericCellValue());
				}
			}
		}
		
		return valueseq;
	}

	
	
	public static void main(String[] args) throws Exception {
		
//		double[] test_unlabel_ratio = new double[26];
//		for(int i=0;i<test_unlabel_ratio.length;i++)
//			test_unlabel_ratio[i]=((double)(50+2*i))/((double)100);

		double[] test_unlabel_ratio = new double[25];
		for(int i=0;i<test_unlabel_ratio.length;i++)
			test_unlabel_ratio[i]=((double)(50+2*i))/((double)100);

		String[] datastr={"bridges_version2"};

		for(int i=0;i<datastr.length;i++){
		String dataname=datastr[i];
		InputStream input = new FileInputStream("ans\\readexcel\\SemiRSFSh_"+dataname+".xlsx");
		
		book=new XSSFWorkbook(input);

//		XSSFCell cell=book.getSheetAt(0).getRow(26).getCell(1);
//		if(cell!=null){
//			System.out.print(cell.getNumericCellValue());
//		}
		
		Vector<Vector<Double>> valueseq=null;
		//1NNPCT
		sheet=book.getSheetAt(0);
		valueseq=readdata(sheet);
		JFreeChart chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Accuracy",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_1nnpct.jpg", 800, 600);

		//cartpct
		sheet=book.getSheetAt(1);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Accuracy",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_cartpct.jpg", 800, 600);

		//AC
		sheet=book.getSheetAt(2);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","AC",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_ac.jpg", 800, 600);

		//NMI
		sheet=book.getSheetAt(3);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","NMI",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_nmi.jpg", 800, 600);
		
		
		//POS
		sheet=book.getSheetAt(4);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Pos",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_pos.jpg", 800, 600);		
		
		//ENTROPY
		sheet=book.getSheetAt(5);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Entropy",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_entropy.jpg", 800, 600);		
		//APPAC
		sheet=book.getSheetAt(6);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Appaccuracy",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_appac.jpg", 800, 600);		
		
		//LENGTH
		sheet=book.getSheetAt(7);
		valueseq=readdata(sheet);
		chart=drawvaluevec(valueseq,methodname,dataname,"Ratio of unlabeled instances","Length",test_unlabel_ratio);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_length.jpg", 800, 600);		
		}
		
	}

}
