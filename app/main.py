from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from .models import Scan, User, Feedback, db
from ml.inference import DiseasePredictor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from functools import wraps
from sqlalchemy import func

main = Blueprint('main', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'Admin':
            flash('Admin access required.')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ['Admin', 'Doctor']: # Admin also passes
            flash('Medical personnel access required.')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

# Dynamically locate model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'skin_disease_model.h5')
predictor = DiseasePredictor(model_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'Admin':
        return redirect(url_for('main.admin_dashboard'))
    
    if current_user.role == 'Doctor':
        # Doctors see all scans
        scans = Scan.query.order_by(Scan.timestamp.desc()).all()
        flash(f'Doctor Access: Viewing {len(scans)} patient records.')
    else:
        # Patients see their own
        scans = current_user.scans

    return render_template('dashboard.html', scans=scans)

@main.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    users = User.query.all()
    feedbacks = Feedback.query.order_by(Feedback.timestamp.desc()).all()
    
    # Analytics
    total_scans = Scan.query.count()
    disease_stats = db.session.query(Scan.disease_name, func.count(Scan.id)).group_by(Scan.disease_name).all()
    
    return render_template('admin_dashboard.html', 
                         users=users, 
                         feedbacks=feedbacks, 
                         total_scans=total_scans, 
                         disease_stats=disease_stats)

@main.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    if current_user.id == user_id:
        flash('Cannot delete yourself.')
        return redirect(url_for('main.admin_dashboard'))
        
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.name} deleted.')
    return redirect(url_for('main.admin_dashboard'))

@main.route('/admin/user/role/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def update_role(user_id):
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    if new_role not in ['Admin', 'Doctor', 'Patient']:
        flash('Invalid role.')
    else:
        user.role = new_role
        db.session.commit()
        flash(f'User {user.name} role updated to {new_role}.')
    return redirect(url_for('main.admin_dashboard'))

@main.route('/admin/train', methods=['POST'])
@login_required
@admin_required
def train_model():
    if 'model_file' not in request.files:
        flash('No file uploaded.')
        return redirect(url_for('main.admin_dashboard'))
    
    file = request.files['model_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('main.admin_dashboard'))
        
    if file and file.filename.endswith('.h5'):
        try:
            # Overwrite existing model
            file.save(model_path)
            # Reload predictor resources
            predictor.load_resources()
            flash('Model updated successfully.')
        except Exception as e:
            flash(f'Error updating model: {e}')
    else:
        flash('Invalid file type. Please upload a .h5 file.')
        
    return redirect(url_for('main.admin_dashboard'))

@main.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            new_feedback = Feedback(user_id=current_user.id, message=message)
            db.session.add(new_feedback)
            db.session.commit()
            flash('Thank you for your feedback!')
            return redirect(url_for('main.dashboard'))
    return render_template('feedback.html')

@main.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run Inference
            symptoms = request.form.get('symptoms')
            age = request.form.get('age')
            gender = request.form.get('gender')
            result = predictor.predict(filepath, symptoms=symptoms)
            
            # Save to DB
            new_scan = Scan(
                user_id=current_user.id,
                image_path=filename,
                disease_name=result['disease'],
                confidence=result['confidence'],
                symptoms=symptoms,
                patient_age=age,
                patient_gender=gender
            )
            db.session.add(new_scan)
            db.session.commit()

            return redirect(url_for('main.result', scan_id=new_scan.id))

    return render_template('detect.html')

@main.route('/result/<int:scan_id>')
@login_required
def result(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    # Check if user is owner OR has valid role
    if scan.user_id != current_user.id and current_user.role not in ['Admin', 'Doctor']:
        flash('Access denied')
        return redirect(url_for('main.dashboard'))
    
    # Fetch info using robust method
    info = predictor.get_disease_info(scan.disease_name)
    
    return render_template('result.html', scan=scan, info=info)

@main.route('/download/<int:scan_id>')
@login_required
def download_report(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and current_user.role not in ['Admin', 'Doctor']:
        return redirect(url_for('main.dashboard'))

    info = predictor.get_disease_info(scan.disease_name)

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    p.setFont("Helvetica-Bold", 24)
    p.drawString(50, height - 50, "Skin Disease Detection Report")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Date: {scan.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(50, height - 100, f"Patient: {current_user.name}")
    p.drawString(50, height - 120, f"Age: {scan.patient_age if scan.patient_age else 'N/A'}")
    p.drawString(300, height - 120, f"Gender: {scan.patient_gender if scan.patient_gender else 'N/A'}")
    
    # Result
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, height - 150, f"Diagnosis: {scan.disease_name}")
    p.drawString(50, height - 175, f"Confidence: {scan.confidence}%")

    if scan.symptoms:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height - 210, "Reported Symptoms:")
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 230, scan.symptoms)
        y_offset = 260
    else:
        y_offset = 210
    
    # Info
    y = height - y_offset
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Description:")
    p.setFont("Helvetica", 12)
    p.drawString(50, y - 20, info.get('description', 'N/A'))
    
    y -= 60
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Dietary Advice:")
    p.setFont("Helvetica", 12)
    p.drawString(50, y - 20, "Eat: " + ", ".join(info.get('diet', {}).get('eat', [])))
    p.drawString(50, y - 40, "Avoid: " + ", ".join(info.get('diet', {}).get('avoid', [])))
    
    y -= 80
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, "Medicine / Treatment:")
    p.setFont("Helvetica", 12)
    p.drawString(50, y - 20, ", ".join(info.get('medicine', [])))
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f'report_{scan.id}.pdf')
