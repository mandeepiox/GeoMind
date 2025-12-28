from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from datetime import datetime
import io
import base64
from typing import Dict, Any

class PDFReportGenerator:
    """Generate professional PDF reports for soil analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Info text
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#475569'),
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
    
    def _create_header(self, canvas_obj, doc):
        """Create header for each page"""
        canvas_obj.saveState()
        
        # Header background
        canvas_obj.setFillColor(colors.HexColor('#2563eb'))
        canvas_obj.rect(0, letter[1] - 0.75*inch, letter[0], 0.75*inch, fill=1, stroke=0)
        
        # Logo/Icon
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont('Helvetica-Bold', 24)
        canvas_obj.drawString(0.75*inch, letter[1] - 0.5*inch, "🌾")
        
        # Title
        canvas_obj.setFont('Helvetica-Bold', 16)
        canvas_obj.drawString(1.25*inch, letter[1] - 0.45*inch, "Punjab Soil Analyzer")
        
        # Subtitle
        canvas_obj.setFont('Helvetica', 10)
        canvas_obj.drawString(1.25*inch, letter[1] - 0.6*inch, "Professional Soil Analysis Report")
        
        canvas_obj.restoreState()
    
    def _create_footer(self, canvas_obj, doc):
        """Create footer for each page"""
        canvas_obj.saveState()
        
        # Footer line
        canvas_obj.setStrokeColor(colors.HexColor('#e2e8f0'))
        canvas_obj.line(0.75*inch, 0.75*inch, letter[0] - 0.75*inch, 0.75*inch)
        
        # Page number
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.setFillColor(colors.HexColor('#64748b'))
        page_num = f"Page {doc.page}"
        canvas_obj.drawRightString(letter[0] - 0.75*inch, 0.5*inch, page_num)
        
        # Generation date
        canvas_obj.drawString(0.75*inch, 0.5*inch, 
                             f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        
        canvas_obj.restoreState()
    
    def _create_pie_chart(self, gravel, sand, silt_clay):
        """Create pie chart for soil composition"""
        drawing = Drawing(400, 300)
        
        pie = Pie()
        pie.x = 100
        pie.y = 50
        pie.width = 200
        pie.height = 200
        pie.data = [gravel, sand, silt_clay]
        pie.labels = ['Gravel', 'Sand', 'Silt & Clay']
        pie.slices.strokeWidth = 2
        pie.slices.strokeColor = colors.white
        
        # Colors matching the web interface
        pie.slices[0].fillColor = colors.HexColor('#8b4513')  # Brown
        pie.slices[1].fillColor = colors.HexColor('#f4c430')  # Golden
        pie.slices[2].fillColor = colors.HexColor('#d2691e')  # Chocolate
        
        # Add legend
        legend = Legend()
        legend.x = 320
        legend.y = 150
        legend.columnMaximum = 3
        legend.alignment = 'right'
        legend.colorNamePairs = [
            (colors.HexColor('#8b4513'), f'Gravel: {gravel:.1f}%'),
            (colors.HexColor('#f4c430'), f'Sand: {sand:.1f}%'),
            (colors.HexColor('#d2691e'), f'Silt & Clay: {silt_clay:.1f}%')
        ]
        legend.fontSize = 10
        legend.fontName = 'Helvetica'
        
        drawing.add(pie)
        drawing.add(legend)
        
        return drawing
    
    def _create_bar_chart(self, gravel, sand, silt_clay):
        """Create bar chart for soil composition"""
        drawing = Drawing(400, 300)
        
        bar = VerticalBarChart()
        bar.x = 50
        bar.y = 50
        bar.width = 300
        bar.height = 200
        bar.data = [[gravel, sand, silt_clay]]
        bar.categoryAxis.categoryNames = ['Gravel', 'Sand', 'Silt & Clay']
        bar.valueAxis.valueMin = 0
        bar.valueAxis.valueMax = 100
        bar.valueAxis.valueStep = 25
        bar.bars[0].fillColor = colors.HexColor('#2563eb')
        bar.bars.strokeColor = colors.white
        bar.bars.strokeWidth = 2
        
        # Labels
        bar.categoryAxis.labels.fontName = 'Helvetica'
        bar.categoryAxis.labels.fontSize = 10
        bar.valueAxis.labels.fontName = 'Helvetica'
        bar.valueAxis.labels.fontSize = 9
        
        drawing.add(bar)
        
        return drawing
    
    def _determine_soil_type(self, gravel, sand, silt_clay):
        """Determine soil type based on composition"""
        total = gravel + sand + silt_clay
        normalized_sand = (sand / total) * 100
        normalized_silt = (silt_clay / total) * 100
        
        if normalized_sand > 85:
            return 'Sandy Soil', [
                'High drainage capacity',
                'Low water retention',
                'Easy to work with',
                'Low nutrient retention',
                'Good aeration'
            ]
        elif normalized_silt > 80:
            return 'Silty Soil', [
                'Moderate drainage',
                'Good water retention',
                'Prone to compaction',
                'High fertility potential',
                'Smooth texture'
            ]
        elif normalized_silt < 40 and normalized_sand > 40:
            return 'Clay Soil', [
                'Poor drainage',
                'High water retention',
                'Difficult to work when wet',
                'High nutrient retention',
                'Poor aeration'
            ]
        else:
            return 'Loam Soil', [
                'Balanced drainage',
                'Good water retention',
                'Excellent workability',
                'High fertility',
                'Ideal for most crops'
            ]
    
    def generate_report(self, prediction_data: Dict[str, Any], output_path: str = None) -> bytes:
        """
        Generate PDF report from prediction data
        
        Args:
            prediction_data: Dictionary containing prediction results
            output_path: Optional file path to save PDF
        
        Returns:
            bytes: PDF file content
        """
        # Create buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        # Container for PDF elements
        story = []
        
        # Extract data
        input_data = prediction_data.get('input', {})
        predictions = prediction_data.get('predictions', {})
        
        latitude = input_data.get('latitude', 0)
        longitude = input_data.get('longitude', 0)
        depth = input_data.get('depth', 0)
        
        # Title
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Soil Analysis Report", self.styles['CustomTitle']))
        story.append(Paragraph(
            f"Location: {latitude:.6f}°N, {longitude:.6f}°E | Depth: {depth}m",
            self.styles['CustomSubtitle']
        ))
        
        # Location Information Table
        story.append(Paragraph("Location Details", self.styles['SectionHeading']))
        
        location_data = [
            ['Parameter', 'Value'],
            ['Latitude', f"{latitude:.6f}°N"],
            ['Longitude', f"{longitude:.6f}°E"],
            ['Depth', f"{depth} meters"],
            ['Analysis Date', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Time', datetime.now().strftime('%I:%M %p')]
        ]
        
        location_table = Table(location_data, colWidths=[2.5*inch, 3*inch])
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        
        story.append(location_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Soil Composition Analysis
        gravel = predictions.get('Gravel', {}).get('value', 0)
        sand = predictions.get('Sand', {}).get('value', 0)
        silt_clay = predictions.get('Silt & Clay', {}).get('value', 0)
        
        if gravel > 0 or sand > 0 or silt_clay > 0:
            story.append(Paragraph("Soil Composition Analysis", self.styles['SectionHeading']))
            
            # Determine soil type
            soil_type, characteristics = self._determine_soil_type(gravel, sand, silt_clay)
            
            story.append(Paragraph(
                f"<b>Soil Type:</b> {soil_type}",
                self.styles['InfoText']
            ))
            story.append(Spacer(1, 0.2*inch))
            
            # Add pie chart
            story.append(Paragraph("Particle Distribution (Pie Chart)", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            story.append(self._create_pie_chart(gravel, sand, silt_clay))
            story.append(Spacer(1, 0.3*inch))
            
            # Add bar chart
            story.append(Paragraph("Composition Breakdown (Bar Chart)", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            story.append(self._create_bar_chart(gravel, sand, silt_clay))
            story.append(Spacer(1, 0.3*inch))
            
            # Soil characteristics
            story.append(Paragraph("Soil Characteristics:", self.styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            char_list = "<br/>".join([f"• {char}" for char in characteristics])
            story.append(Paragraph(char_list, self.styles['InfoText']))
            story.append(Spacer(1, 0.2*inch))
        
        # Page break before detailed properties
        story.append(PageBreak())
        
        # Detailed Property Analysis
        story.append(Paragraph("Detailed Property Analysis", self.styles['SectionHeading']))
        
        # Properties table
        property_data = [['Property', 'Value', 'Unit', 'R² Score', 'Confidence']]
        
        property_icons = {
            'N-value': 'N-value',
            'Bulk Density': 'Bulk Density',
            'Cohesion': 'Cohesion',
            'Shear angle': 'Shear Angle',
            'Gravel': 'Gravel',
            'Sand': 'Sand',
            'Silt & Clay': 'Silt & Clay'
        }
        
        for prop_name, prop_data in predictions.items():
            if prop_data.get('value') is not None:
                display_name = property_icons.get(prop_name, prop_name)
                value = f"{prop_data['value']:.2f}"
                unit = prop_data.get('unit', '')
                r2 = f"{prop_data['r2_score']:.3f}"
                confidence = prop_data['confidence']
                
                property_data.append([display_name, value, unit, r2, confidence])
        
        properties_table = Table(property_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        properties_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        
        story.append(properties_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Confidence Legend
        story.append(Paragraph("Confidence Level Guide:", self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        confidence_text = """
        <b>High Confidence (R² > 0.8):</b> Very reliable predictions<br/>
        <b>Medium Confidence (R² 0.6-0.8):</b> Good estimates, use with some caution<br/>
        <b>Low Confidence (R² < 0.6):</b> Use cautiously, field verification recommended
        """
        story.append(Paragraph(confidence_text, self.styles['InfoText']))
        
        # Disclaimer
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Disclaimer & Recommendations", self.styles['SectionHeading']))
        
        disclaimer = """
        This report is generated using machine learning predictions based on geographical coordinates 
        and depth. While our models achieve high accuracy, field verification is recommended for critical 
        applications. The predictions are based on historical soil survey data from the Punjab region. 
        For construction or agricultural planning, please consult with local soil experts and conduct 
        on-site testing to confirm these predictions.
        """
        story.append(Paragraph(disclaimer, self.styles['InfoText']))
        
        # Build PDF with custom headers and footers
        doc.build(story, onFirstPage=self._create_header, onLaterPages=self._create_header,
                 canvasmaker=lambda *args, **kwargs: self._create_footer_canvas(*args, **kwargs))
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_content)
        
        return pdf_content
    
    def _create_footer_canvas(self, *args, **kwargs):
        """Create canvas with footer"""
        c = canvas.Canvas(*args, **kwargs)
        original_showPage = c.showPage
        
        def custom_showPage():
            self._create_footer(c, c)
            original_showPage()
        
        c.showPage = custom_showPage
        return c


# Utility function for easy use
def generate_soil_report(prediction_data: Dict[str, Any], output_path: str = None) -> bytes:
    """
    Convenience function to generate PDF report
    
    Args:
        prediction_data: Prediction results from API
        output_path: Optional file path to save
    
    Returns:
        bytes: PDF content
    """
    generator = PDFReportGenerator()
    return generator.generate_report(prediction_data, output_path)
