"""
Script para organizar dataset de Kaggle
Convierte de estructura LostCat-PS/LostCat-PSC a estructura simple
"""

import os
import shutil
from pathlib import Path

class DatasetOrganizer:
    """Organiza dataset de Kaggle"""
    
    def __init__(self, base_path='./'):
        self.base_path = base_path
        self.data_path = './data'
        
    def organize(self):
        """Organiza las imágenes"""
        
        print("\n" + "="*70)
        print("ORGANIZANDO DATASET DE KAGGLE")
        print("="*70 + "\n")
        
        # Crear estructura
        pet_dir = os.path.join(self.data_path, 'pet')
        stray_dir = os.path.join(self.data_path, 'stray')
        community_dir = os.path.join(self.data_path, 'community')
        
        os.makedirs(pet_dir, exist_ok=True)
        os.makedirs(stray_dir, exist_ok=True)
        os.makedirs(community_dir, exist_ok=True)
        
        print(f"[ORGANIZE] Carpetas creadas:")
        print(f"  - {pet_dir}")
        print(f"  - {stray_dir}")
        print(f"  - {community_dir}\n")
        
        # Buscar y copiar imágenes de LostCat-PS
        lost_cat_ps_1 = os.path.join(self.base_path, 'LostCat-PS', 'LostCat-PS')
        lost_cat_ps_2 = os.path.join(self.base_path, 'LostCat-PS')
        
        if os.path.exists(lost_cat_ps_1):
            print(f"[ORGANIZE] Procesando LostCat-PS/LostCat-PS")
            self._copy_images(os.path.join(lost_cat_ps_1, 'pet'), pet_dir, 'LostCat-PS/pet')
            self._copy_images(os.path.join(lost_cat_ps_1, 'stray'), stray_dir, 'LostCat-PS/stray')
        elif os.path.exists(lost_cat_ps_2):
            print(f"[ORGANIZE] Procesando LostCat-PS")
            self._copy_images(os.path.join(lost_cat_ps_2, 'pet'), pet_dir, 'LostCat-PS/pet')
            self._copy_images(os.path.join(lost_cat_ps_2, 'stray'), stray_dir, 'LostCat-PS/stray')
        else:
            print(f"[ORGANIZE] ⚠️  LostCat-PS no encontrado")
        
        # Buscar y copiar imágenes de LostCat-PSC
        lost_cat_psc_1 = os.path.join(self.base_path, 'LostCat-PSC', 'LostCat-PSC')
        lost_cat_psc_2 = os.path.join(self.base_path, 'LostCat-PSC')
        
        if os.path.exists(lost_cat_psc_1):
            print(f"[ORGANIZE] Procesando LostCat-PSC/LostCat-PSC")
            self._copy_images(os.path.join(lost_cat_psc_1, 'pet'), pet_dir, 'LostCat-PSC/pet')
            self._copy_images(os.path.join(lost_cat_psc_1, 'stray'), stray_dir, 'LostCat-PSC/stray')
            self._copy_images(os.path.join(lost_cat_psc_1, 'community'), community_dir, 'LostCat-PSC/community')
        elif os.path.exists(lost_cat_psc_2):
            print(f"[ORGANIZE] Procesando LostCat-PSC")
            self._copy_images(os.path.join(lost_cat_psc_2, 'pet'), pet_dir, 'LostCat-PSC/pet')
            self._copy_images(os.path.join(lost_cat_psc_2, 'stray'), stray_dir, 'LostCat-PSC/stray')
            self._copy_images(os.path.join(lost_cat_psc_2, 'community'), community_dir, 'LostCat-PSC/community')
        else:
            print(f"[ORGANIZE] ⚠️  LostCat-PSC no encontrado")
        
        print("\n[ORGANIZE] ✓ Dataset organizado\n")
        self.verify()
    
    def _copy_images(self, src, dst, label):
        """Copia imágenes de origen a destino"""
        
        if not os.path.exists(src):
            print(f"[ORGANIZE] ⚠️  Carpeta no encontrada: {label}")
            return
        
        images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            print(f"[ORGANIZE] Copiando {len(images)} imágenes de {label}")
            for img in images:
                src_path = os.path.join(src, img)
                dst_path = os.path.join(dst, img)
                
                # No copiar si ya existe
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"[ORGANIZE] Error copiando {img}: {e}")
        else:
            print(f"[ORGANIZE] ⚠️  No hay imágenes en: {label}")
    
    def verify(self):
        """Verifica que el dataset esté completo"""
        
        print("="*70)
        print("VERIFICACIÓN DEL DATASET")
        print("="*70 + "\n")
        
        pet_dir = os.path.join(self.data_path, 'pet')
        stray_dir = os.path.join(self.data_path, 'stray')
        community_dir = os.path.join(self.data_path, 'community')
        
        pet_count = len([f for f in os.listdir(pet_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        stray_count = len([f for f in os.listdir(stray_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        community_count = len([f for f in os.listdir(community_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Gatos Mascota (pet): {pet_count} imágenes")
        print(f"Gatos Callejeros (stray): {stray_count} imágenes")
        print(f"Gatos Comunidad (community): {community_count} imágenes")
        print(f"Total: {pet_count + stray_count + community_count} imágenes\n")
        
        if pet_count == 0 or stray_count == 0:
            print("[ORGANIZE] ⚠️  Advertencia: Dataset incompleto\n")
            return False
        
        print("[ORGANIZE] ✓ Dataset verificado correctamente\n")
        print("📁 Estructura final:")
        print("  data/")
        print(f"    ├── pet/        ({pet_count} imágenes)")
        print(f"    ├── stray/      ({stray_count} imágenes)")
        print(f"    └── community/  ({community_count} imágenes)\n")
        
        return True

if __name__ == "__main__":
    # Usar directorio actual
    organizer = DatasetOrganizer('./')
    organizer.organize()